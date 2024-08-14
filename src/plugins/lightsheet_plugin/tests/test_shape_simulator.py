import unittest
import numpy as np
from biological_simulation import Sphere, Cube, Cylinder, Torus, Cone

class TestShapes(unittest.TestCase):
    def setUp(self):
        self.position = [0, 0, 0]
        self.size = 10
        self.color = [1, 1, 1]
        self.resolution = 10

    def test_sphere(self):
        sphere = Sphere(self.position, self.size, self.color)
        points = sphere.get_volume_points(self.resolution)
        self.assertEqual(points.shape[1], 3)  # 3D points
        self.assertGreater(len(points), 0)
        # Check if points are within the sphere
        distances = np.linalg.norm(points - self.position, axis=1)
        self.assertTrue(np.all(distances <= self.size/2 + 1e-10))  # Added small tolerance)

    def test_cube(self):
        cube = Cube(self.position, self.size, self.color)
        points = cube.get_volume_points(self.resolution)
        self.assertEqual(points.shape[1], 3)  # 3D points
        self.assertGreater(len(points), 0)
        # Check if points are within the cube
        self.assertTrue(np.all(np.abs(points - self.position) <= self.size/2))

    def test_cylinder(self):
        cylinder = Cylinder(self.position, self.size, self.color)
        points = cylinder.get_volume_points(self.resolution)
        self.assertEqual(points.shape[1], 3)  # 3D points
        self.assertGreater(len(points), 0)
        # Check if points are within the cylinder
        distances = np.linalg.norm(points[:, :2] - self.position[:2], axis=1)
        heights = np.abs(points[:, 2] - self.position[2])
        self.assertTrue(np.all(distances <= self.size/2))
        self.assertTrue(np.all(heights <= self.size/2))

    def test_torus(self):
        torus = Torus(self.position, self.size, self.color)
        points = torus.get_volume_points(self.resolution)
        self.assertEqual(points.shape[1], 3)  # 3D points
        self.assertGreater(len(points), 0)
        # Check if points are within the torus
        R = self.size / 2
        r = R / 4
        distances = np.linalg.norm(points[:, :2] - self.position[:2], axis=1)
        self.assertTrue(np.all((R-r <= distances) & (distances <= R+r)))
        self.assertTrue(np.all(np.abs(points[:, 2] - self.position[2]) <= r))

    def test_cone(self):
        cone = Cone(self.position, self.size, self.color)
        points = cone.get_volume_points(self.resolution)
        self.assertEqual(points.shape[1], 3)  # 3D points
        self.assertGreater(len(points), 0)
        # Check if points are within the cone
        heights = points[:, 2] - self.position[2]
        radii = np.linalg.norm(points[:, :2] - self.position[:2], axis=1)
        self.assertTrue(np.all((0 <= heights) & (heights <= self.size)))
        self.assertTrue(np.all(radii <= (self.size/2) * (1 - heights/self.size) + 1e-10))  # Added small tolerance

    def test_rotation(self):
        for ShapeClass in [Sphere, Cube, Cylinder, Torus, Cone]:
            shape = ShapeClass(self.position, self.size, self.color)
            original_vertices = shape.get_vertices()
            rotated_shape = shape.rotate([90, 45, 30])
            rotated_vertices = rotated_shape.get_vertices()
            self.assertFalse(np.allclose(original_vertices, rotated_vertices))

    def test_vertices_and_faces(self):
        for ShapeClass in [Sphere, Cube, Cylinder, Torus, Cone]:
            shape = ShapeClass(self.position, self.size, self.color)
            vertices = shape.get_vertices()
            faces = shape.get_faces()
            self.assertIsInstance(vertices, np.ndarray, f"{ShapeClass.__name__} vertices should be a numpy array")
            self.assertIsInstance(faces, np.ndarray, f"{ShapeClass.__name__} faces should be a numpy array")
            self.assertEqual(vertices.shape[1], 3, f"{ShapeClass.__name__} vertices should be 3D")
            if faces.size > 0:
                self.assertEqual(faces.shape[1], 3, f"{ShapeClass.__name__} faces should be triangular")
            else:
                self.assertEqual(faces.shape, (0, 3), f"{ShapeClass.__name__} faces should be shaped (0, 3) when empty")

    def test_rotation(self):
        for ShapeClass in [Sphere, Cube, Cylinder, Torus, Cone]:
            shape = ShapeClass(self.position, self.size, self.color)
            original_vertices = shape.get_vertices()
            shape.rotate([90, 45, 30])
            rotated_vertices = shape.get_vertices()
            self.assertFalse(np.allclose(original_vertices, rotated_vertices))


if __name__ == '__main__':
    unittest.main()
