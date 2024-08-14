import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import struct
import zlib
from typing import List, Tuple
import re

class SISImporter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata = {}
        self.raw_data = b''
        self.xml_data = ''
        self.images = []
        self.file_path = file_path
        self.metadata = {}
        self.raw_data = b''
        self.image_data_start = 0
        self.image_data = b''

    def read_file(self):
        with open(self.file_path, 'rb') as f:
            self.raw_data = f.read()
        print(f"Read {len(self.raw_data)} bytes from file")

    def extract_xml(self):
        xml_start = self.raw_data.find(b'<Header>')
        xml_end = self.raw_data.find(b'</Header>') + len(b'</Header>')
        if xml_start == -1 or xml_end == -1:
            raise ValueError("Could not find XML content in file")
        self.xml_data = self.raw_data[xml_start:xml_end].decode('utf-8', errors='ignore')
        print("First 500 characters of XML data:")
        print(self.xml_data[:500])

    def parse_metadata(self):
        root = ET.fromstring(self.xml_data)
        self.metadata = self.xml_to_dict(root)

        print("Raw metadata structure:")
        self.print_dict(self.metadata)

        # Extract key metadata
        try:
            scope = self.metadata['Scopes']['Scope']
            self.metadata['channels'] = int(scope['Channels']['@number'])

            # Handle the case where 'Plane' might be a dict instead of a list
            planes = scope['Ticks']['Tick']['Plane']
            if isinstance(planes, dict):
                planes = [planes]
            elif isinstance(planes, list):
                pass
            else:
                raise ValueError(f"Unexpected type for planes: {type(planes)}")

            self.metadata['z_planes'] = len(planes)
            self.metadata['time_points'] = int(scope['Ticks']['@number'])

            # Extract dimensions from the original XML string
            rect_match = re.search(r'<Rect.*?width="(\d+)".*?height="(\d+)"', self.xml_data)
            if rect_match:
                self.metadata['width'] = int(rect_match.group(1))
                self.metadata['height'] = int(rect_match.group(2))
            else:
                raise ValueError("Could not find width and height in XML data")

            tiles_match = re.search(r'<Tiles.*?width="(\d+)".*?height="(\d+)".*?tileNumX="(\d+)".*?tileNumY="(\d+)"', self.xml_data)
            if tiles_match:
                self.metadata['tile_width'] = int(tiles_match.group(1))
                self.metadata['tile_height'] = int(tiles_match.group(2))
                self.metadata['tile_num_x'] = int(tiles_match.group(3))
                self.metadata['tile_num_y'] = int(tiles_match.group(4))
            else:
                raise ValueError("Could not find tile information in XML data")

        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            print("Metadata structure:")
            self.print_dict(self.metadata)
            raise

        print("Extracted metadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")


    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text or None
            else:
                result[child.tag] = self.xml_to_dict(child)
        result.update(('@' + k, v) for k, v in element.attrib.items())
        return result


    def print_dict(self, d, indent=0):
        for key, value in d.items():
            print('  ' * indent + str(key))
            if isinstance(value, dict):
                self.print_dict(value, indent+1)
            else:
                print('  ' * (indent+1) + str(value))

    def locate_image_data(self):
        # Assuming image data starts after XML
        xml_end = self.raw_data.find(b'</Header>') + len(b'</Header>')
        return xml_end


    def parse_images(self):
        self.image_data_start = self.raw_data.find(b'\x00\x00\x00\x00', len(self.xml_data))
        if self.image_data_start == -1:
            print("Could not locate image data")
            return

        self.image_data = self.raw_data[self.image_data_start:]
        print(f"Potential image data starts at byte {self.image_data_start}")
        print(f"First 100 bytes of image data: {self.image_data[:100].hex()}")

        # Parse the first few integers to see if there's a header
        header_ints = struct.unpack('>10I', self.image_data[:40])
        print(f"First 10 integers (big-endian): {header_ints}")

        # Try parsing as different data types
        self.uint8_array = np.frombuffer(self.image_data, dtype=np.uint8)

        # Ensure the data length is even for 16-bit integers
        even_length = (len(self.image_data) // 2) * 2
        self.uint16_be = np.frombuffer(self.image_data[:even_length], dtype='>u2')
        self.uint16_le = np.frombuffer(self.image_data[:even_length], dtype='<u2')

        # Ensure the data length is a multiple of 4 for 32-bit integers and floats
        multiple_of_four_length = (len(self.image_data) // 4) * 4
        self.uint32_be = np.frombuffer(self.image_data[:multiple_of_four_length], dtype='>u4')
        self.float32_be = np.frombuffer(self.image_data[:multiple_of_four_length], dtype='>f4')

        print(f"uint8 array: shape={self.uint8_array.shape}, min={np.min(self.uint8_array)}, max={np.max(self.uint8_array)}")
        print(f"uint16 big-endian: shape={self.uint16_be.shape}, min={np.min(self.uint16_be)}, max={np.max(self.uint16_be)}")
        print(f"uint16 little-endian: shape={self.uint16_le.shape}, min={np.min(self.uint16_le)}, max={np.max(self.uint16_le)}")
        print(f"uint32 big-endian: shape={self.uint32_be.shape}, min={np.min(self.uint32_be)}, max={np.max(self.uint32_be)}")
        print(f"float32 big-endian: shape={self.float32_be.shape}, min={np.min(self.float32_be)}, max={np.max(self.float32_be)}")

        # Analyze the first few bytes in more detail
        print("\nDetailed analysis of first 40 bytes:")
        for i in range(0, 40, 4):
            bytes_4 = self.image_data[i:i+4]
            uint32 = struct.unpack('>I', bytes_4)[0]
            float32 = struct.unpack('>f', bytes_4)[0]
            print(f"Bytes {i}-{i+3}: {bytes_4.hex()} | uint32: {uint32} | float32: {float32}")

        # Analyze float32 data more closely
        float32_finite = self.float32_be[np.isfinite(self.float32_be)]
        print(f"float32 (finite values only): shape={float32_finite.shape}, min={np.min(float32_finite)}, max={np.max(float32_finite)}")

        # Check for potential tiled structure
        tile_size = self.metadata['tile_width'] * self.metadata['tile_height']
        if len(self.uint16_be) >= tile_size:
            first_tile = self.uint16_be[:tile_size].reshape(self.metadata['tile_height'], self.metadata['tile_width'])
            print(f"First tile statistics: min={np.min(first_tile)}, max={np.max(first_tile)}, mean={np.mean(first_tile)}")


        # Analyze uint16 data to find the potential start of image data
        diff = np.diff(self.uint16_be)
        potential_start = np.where(np.abs(diff) > 1000)[0][0] + 1
        print(f"Potential start of image data in uint16 array: index {potential_start}")

        # Analyze the distribution of values in uint16 data after the potential start
        image_data = self.uint16_be[potential_start:]
        unique, counts = np.unique(image_data, return_counts=True)
        print(f"Number of unique values in uint16 image data: {len(unique)}")
        print(f"Top 10 most common values: {unique[np.argsort(counts)[-10:][::-1]]}")

        # Calculate statistics for the potential image data
        print(f"Image data statistics: min={np.min(image_data)}, max={np.max(image_data)}, mean={np.mean(image_data)}")


    def display_images(self):
        image_data = self.uint16_be[999:]  # Start from the identified data start point

        tile_width = self.metadata['tile_width']
        tile_height = self.metadata['tile_height']
        tile_num_x = self.metadata['tile_num_x']
        tile_num_y = self.metadata['tile_num_y']
        channels = self.metadata['channels']
        z_planes = self.metadata['z_planes']

        full_image_width = tile_width * tile_num_x
        full_image_height = tile_height * tile_num_y
        pixels_per_full_image = full_image_width * full_image_height

        # Calculate the number of complete full images (considering all channels and z-planes)
        num_complete_images = len(image_data) // (pixels_per_full_image * channels * z_planes)
        print(f"Number of complete full images: {num_complete_images}")

        if num_complete_images > 0:
            for c in range(channels):
                fig, axes = plt.subplots(1, tile_num_x, figsize=(20, 7))
                fig.suptitle(f"Channel {c+1} - Tiles")

                for t in range(tile_num_x):
                    start = (c * pixels_per_full_image) + (t * tile_width * tile_height)
                    end = start + (tile_width * tile_height)
                    tile_data = image_data[start:end].reshape(tile_height, tile_width)

                    if tile_num_x == 1:
                        ax = axes
                    else:
                        ax = axes[t]

                    im = ax.imshow(tile_data, cmap='viridis')
                    ax.set_title(f"Tile {t+1}")
                    plt.colorbar(im, ax=ax)

                plt.tight_layout()
                plt.show()

            # Display full reconstructed image for each channel
            fig, axes = plt.subplots(1, channels, figsize=(20, 10))
            fig.suptitle("Reconstructed Full Images")

            for c in range(channels):
                full_image = np.zeros((full_image_height, full_image_width), dtype=np.uint16)
                for t in range(tile_num_x):
                    start = (c * pixels_per_full_image) + (t * tile_width * tile_height)
                    end = start + (tile_width * tile_height)
                    tile_data = image_data[start:end].reshape(tile_height, tile_width)
                    full_image[:, t*tile_width:(t+1)*tile_width] = tile_data

                if channels == 1:
                    ax = axes
                else:
                    ax = axes[c]

                im = ax.imshow(full_image, cmap='viridis')
                ax.set_title(f"Channel {c+1}")
                plt.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.show()

        # Display a larger portion of the data as a 2D image
        plt.figure(figsize=(15, 15))
        plt.imshow(image_data[:90000].reshape(300, 300), cmap='viridis', aspect='auto')
        plt.title("First 90000 values of image data as 2D image")
        plt.colorbar()
        plt.show()


# Usage

importer = SISImporter('/Users/george/Desktop/biologicalSimulator/cropped_synapses.sis')
importer.read_file()  # Assume this method exists and reads the file into self.raw_data
importer.extract_xml()  # Assume this method exists and extracts XML into self.xml_data
importer.parse_metadata()  # Assume this method exists and parses metadata
importer.parse_images()
importer.display_images()

#     print(f"Raw data size: {len(importer.raw_data)} bytes")
#     print(f"Number of images parsed: {importer.images.shape[0] if len(importer.images) > 0 else 0}")
# except Exception as e:
#     print(f"An error occurred: {str(e)}")
#     import traceback
#     traceback.print_exc()



