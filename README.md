# Prediction API

This is a FastAPI application that provides a prediction API. It takes a `Scene` object as input and returns a `Prediction` object. The application is designed with a clear and scalable architecture, using the Strategy design pattern for the prediction model and logging requests and responses to a MongoDB database.

## Features

-   **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
-   **Pydantic**: Data validation and settings management using Python type annotations.
-   **MongoDB**: A NoSQL database for storing prediction logs.
-   **Strategy Design Pattern**: The prediction service is designed to be extensible, allowing for different prediction models to be used interchangeably.
-   **Swagger UI**: Interactive API documentation and testing interface.

## Project Structure

```
.
├── app
│   ├── db
│   │   ├── database.py
│   │   └── __init__.py
│   ├── models
│   │   ├── prediction.py
│   │   ├── scene.py
│   │   └── __init__.py
│   ├── routers
│   │   ├── prediction.py
│   │   └── __init__.py
│   ├── services
│   │   ├── dummy_prediction_service.py
│   │   ├── logging_service.py
│   │   ├── prediction_service.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── main.py
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

-   Python 3.7+
-   MongoDB

### Installation

1.  Clone the repository:

    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2.  Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Make sure your MongoDB instance is running.

2.  Run the application:

    ```bash
    python main.py
    ```

The application will be available at `http://localhost:8000`.

## API Documentation

Once the application is running, you can access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`. This interface allows you to explore the API endpoints, view the request and response models, and test the API directly from your browser.

## .gitignore

This project includes a `.gitignore` file to exclude unnecessary files and directories from version control. This helps maintain a clean and organized repository by ignoring byte-code, virtual environments, and IDE-specific configuration files. Please ensure that your local development environment respects this file to avoid committing sensitive or irrelevant data. 