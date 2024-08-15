from PyQt5.QtWidgets import QMessageBox
from particle_analysis import ParticleAnalysisResults
import pyqtgraph as pg
import numpy as np

class ParticleAnalysisOperations:
    def __init__(self, parent):
        self.parent = parent

    def run_particle_analysis(self):
        if self.parent.window_manager.current_window is None:
            QMessageBox.warning(self.parent, "No Image", "Please open an image first.")
            return

        try:
            image = self.parent.window_manager.current_window.image
            analysis_dialog = ParticleAnalysisResults(self.parent, image)
            analysis_dialog.analysisComplete.connect(self.parent.on_particle_analysis_complete)
            analysis_dialog.exec_()
        except Exception as e:
            print(f"Error in particle analysis: {str(e)}")
            QMessageBox.critical(self.parent, "Error", f"Error in particle analysis: {str(e)}")

    def toggle_results_chart(self, checked):
        if self.parent.particle_analysis_results is not None:
            if checked:
                self.show_results_chart()
            else:
                self.hide_results_chart()

    def show_results_chart(self):
        if not hasattr(self.parent, 'results_chart_window'):
            self.parent.results_chart_window = self.create_results_chart_window()
        self.parent.results_chart_window.show()

    def hide_results_chart(self):
        if hasattr(self.parent, 'results_chart_window'):
            self.parent.results_chart_window.hide()

    def create_results_chart_window(self):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableView
        from particle_analysis import PandasModel

        window = QWidget()
        layout = QVBoxLayout()

        # Create table view
        table_view = QTableView()
        model = PandasModel(self.parent.particle_analysis_results)
        table_view.setModel(model)
        layout.addWidget(table_view)

        # Create scatter plot
        scatter_widget = pg.PlotWidget()
        scatter_widget.setLabel('left', 'Y')
        scatter_widget.setLabel('bottom', 'X')
        scatter_widget.setTitle('Particle Locations and Trajectories')

        particles = self.parent.particle_analysis_results
        is_linked = 'particle' in particles.columns

        max_particles_to_plot = 10000

        if is_linked:
            unique_particles = particles['particle'].unique()
            if len(unique_particles) > max_particles_to_plot:
                unique_particles = np.random.choice(unique_particles, max_particles_to_plot, replace=False)

            scatter_plot = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
            for particle_id in unique_particles:
                particle_data = particles[particles['particle'] == particle_id]
                if not particle_data.empty:
                    color = pg.intColor(particle_id, hues=len(unique_particles))
                    x = particle_data['centroid-1'].values
                    y = particle_data['centroid-0'].values
                    scatter_plot.addPoints(x=x, y=y, brush=color)
            scatter_widget.addItem(scatter_plot)
        else:
            if len(particles) > max_particles_to_plot:
                particles = particles.sample(max_particles_to_plot)

            x = particles['centroid-1'].values
            y = particles['centroid-0'].values
            scatter_plot = pg.ScatterPlotItem(x=x, y=y, size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
            scatter_widget.addItem(scatter_plot)

        layout.addWidget(scatter_widget)

        window.setLayout(layout)
        window.setWindowTitle("Particle Analysis Results")
        window.resize(800, 600)

        return window

    def toggle_centroids(self, checked):
        if self.parent.particle_analysis_results is not None:
            current_window = self.parent.window_manager.current_window
            if current_window:
                if checked:
                    self.plot_centroids(current_window)
                else:
                    self.remove_centroids(current_window)

    def remove_centroids(self, window):
        if hasattr(window, 'centroid_items'):
            for item in window.centroid_items:
                window.get_view().removeItem(item)
            window.centroid_items.clear()

    def plot_centroids(self, window):
        if not hasattr(window, 'centroid_items'):
            window.centroid_items = []

        self.remove_centroids(window)

        current_frame = window.get_current_frame()
        frame_particles = self.parent.particle_analysis_results[self.parent.particle_analysis_results['frame'] == current_frame]

        is_trackpy = 'particle' in self.parent.particle_analysis_results.columns

        for _, row in frame_particles.iterrows():
            color = pg.intColor(row['particle'], hues=50, alpha=120) if is_trackpy else pg.mkBrush(255, 0, 0, 120)

            centroid = pg.ScatterPlotItem([row['centroid-1']], [row['centroid-0']], size=10, pen=pg.mkPen(None), brush=color)
            window.get_view().addItem(centroid)
            window.centroid_items.append(centroid)

            if is_trackpy:
                trajectory = self.parent.particle_analysis_results[self.parent.particle_analysis_results['particle'] == row['particle']]
                trajectory = trajectory[trajectory['frame'] <= current_frame]
                if len(trajectory) > 1:
                    trajectory_item = pg.PlotDataItem(trajectory['centroid-1'], trajectory['centroid-0'], pen=color)
                    window.get_view().addItem(trajectory_item)
                    window.centroid_items.append(trajectory_item)

        if hasattr(self.parent, 'particle_count_label'):
            particle_count = len(frame_particles)
            self.parent.particle_count_label.setText(f"Particles in frame: {particle_count}")

    def on_time_slider_changed(self):
        if self.parent.toggle_centroids_button.isChecked():
            current_window = self.parent.window_manager.current_window
            if current_window:
                self.plot_centroids(current_window)
