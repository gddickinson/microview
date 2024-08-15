class MetadataOperations:
    def __init__(self, parent):
        self.parent = parent

    def display_metadata_info(self, metadata):
        info = f"Image dimensions: {metadata['dims']}\n"
        info += f"Image shape: {metadata['shape']}\n"
        info += f"Data type: {metadata['dtype']}\n"

        if metadata['pixel_size_um']:
            info += f"Pixel size: {metadata['pixel_size_um']} µm\n"
        if metadata['time_interval_s']:
            info += f"Time interval: {metadata['time_interval_s']} s\n"
        if metadata['channel_names']:
            info += f"Channels: {', '.join(metadata['channel_names'])}\n"

        self.parent.QMessageBox.information(self.parent, "Image Metadata", info)
