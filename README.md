# Yoga Pose Estimation and Feedback System

A web-based application that assists users in performing yoga poses with real-time video processing and feedback using a webcam. This project leverages pose estimation models to analyze user posture and provide corrective guidance through both visual and auditory feedback.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to help users practice yoga by providing real-time feedback on their poses. Using a webcam, the application captures the user's movements and utilizes TensorFlow-based pose estimation to analyze the accuracy of the poses. The system provides both visual cues on the video feed and auditory instructions to guide users in adjusting their posture.

## Features

- Real-time video processing using webcam.
- Pose estimation and analysis with TensorFlow Movenet model.
- Visual feedback with skeletal overlay and pose accuracy indicators.
- Auditory feedback using pyttsx3 to provide real-time instructions.
- User-friendly web interface with Flask.

## Installation

### Prerequisites

- Python 3.7 or later
- Flask
- OpenCV
- TensorFlow
- Keras
- pyttsx3

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pose estimation model and place it in the appropriate directory:

    ```bash
    # Example command to download the model
    wget https://path-to-your-model/movenet_thunder.tflite -P models/
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Follow the on-screen instructions to select and perform a yoga pose. The system will provide real-time feedback to help you adjust your posture.

## Technologies Used

- **Flask**: Web framework for building the application.
- **OpenCV**: Library for real-time computer vision.
- **TensorFlow & Keras**: Machine learning framework for pose estimation.
- **pyttsx3**: Text-to-speech conversion library.
- **HTML/CSS/JavaScript**: Front-end technologies for building the user interface.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements or new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
