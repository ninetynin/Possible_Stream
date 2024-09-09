## Setup and Installation

1. **Create a Virtual Environment**:
    ```
    python -m venv venv
    venv\Scripts\activate  # for powershell
2. **Install the Required Libraries**:
    ```
    pip install -r requirements.txt
    ```
3. **Setup the database**:
    ```
    path to your python venv exe db/db.py
    ```
4. **Run the server & client**:
    get your ipv4 address from your terminal using `ipconfig`
    then run these commands
    ```
    python3 src/server.py
    ```
    ```
    python3 src/client.py
    ```
5. **plot the results**:
    ```
    python3 src/plot.py
    ```