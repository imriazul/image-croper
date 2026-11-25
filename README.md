# AI Passport Photo Maker

A professional AI-powered desktop application for automatically processing bulk passport photographs.

## Features

- **AI Face Detection** – MediaPipe Face Mesh for precise facial landmark detection
- **Intelligent Cropping** – Automatic passport-dimension cropping based on facial landmarks
- **Background Removal** – Pure white background replacement using rembg (U²-Net)
- **Bulk Processing** – Process hundreds or thousands of photos with multithreaded engine
- **Quality Validation** – Automatic validation with clear rejection reasons
- **Modern UI** – Clean PySide6 desktop interface with live statistics

## Requirements

- Python 3.9 or later
- Windows 10/11 or macOS

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, MediaPipe and rembg will download their AI models automatically.
> This requires an internet connection for the first run only.

### 3. Run the application

```bash
python main.py
```

## Usage

1. **Select Input** – Click "Select Folder" or "Select ZIP" to choose photographs
2. **Configure** – Adjust passport size (45×55mm default) and DPI (300 default)
3. **Process** – Click "Start Processing" and monitor progress
4. **Results** – Open Cropped/Rejected folders to view results

## Output Structure

```
Passport_Photo_Output/
├── Cropped/          # Successfully processed passport photos
│   ├── photo1.jpg
│   └── photo2.jpg
└── Rejected/         # Photos that couldn't be processed
    ├── photo3.png
    └── photo3.png.reason.txt
```

## Supported Formats

- JPG / JPEG
- PNG
- WEBP

## Building Windows Executable

```bash
pip install pyinstaller
pyinstaller passport_photo_maker.spec
```

The executable will be in the `dist/` folder.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| GUI | PySide6 |
| Face Detection | MediaPipe Face Mesh |
| Background Removal | rembg + U²-Net |
| Image Processing | OpenCV + Pillow |
| Packaging | PyInstaller |

## License

All dependencies use permissible open-source licenses. No paid APIs or subscriptions required.
