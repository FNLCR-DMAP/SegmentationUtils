from setuptools import setup

setup(
    name='pyoseg',
    packages=['pyoseg'],
    package_dir={'':'src'},
    version='0.1.0',
    url='git@github.com:FNLCR-DMAP/SegmentationUtils.git',
    author='Hermann Degenhardt',
    author_email='hermann.degenhardt@nih.gov',
    description='Pyoseg is a python open-source library for object detection and segmentation. It includes modules for dataset split, data augmentation and inference analysis.',
    install_requires=[
        'Pillow >= 9.4.0', 'albumentations >= 1.3.1', 'imageio >= 2.29.0',
        'json5 >= 0.9.5', 'matplotlib >= 3.6.3', 'numpy >= 1.22.3',
        'opencv-python >= 4.7.0.72', 'pandas >= 1.5.3', 'pycocotools >= 2.0.6',
        'scikit-image >= 0.20.0', 'scikit-learn >= 1.2.1', 'scipy >= 1.9.1',
        'seaborn >= 0.12.2', 'shapely >= 2.0.1', 'tifffile >= 2023.4.12',
        'pyyaml >=6.0'],
)
