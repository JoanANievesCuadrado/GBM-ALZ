# GB-ALZ

This repository contains the data and algorithms used in the preprint [10.1101/2023.11.23.568350](https://doi.org/10.1101/2023.11.23.568350).

## Repository Contents

- `main.py`: Performs PCA analysis and geometric analyses, creates the 4 main subfigures of the article, and generates tables with the most important genes of PC1 and PC2.
- `make_translate_file.py`: Additional scripts used in the project.
- `mouse.py`: Analyzes mouse data and creates Supplementary Figure 1.
- `data`: Contains the genetic expression data used.
- `Figures and tables`: Contains the tables and figures used in the article and is where all the results of the scripts are stored.
- `requirements.txt`: File with the necessary dependencies to run the scripts.
- `runtime.txt`: File with the runtime environment configuration.

## How to Use This Repository

1. Clone the repository:
    ```bash
    git clone https://github.com/JoanANievesCuadrado/GB_vs_AD.git
    ```
2. Go to the `data` folder and extract the files from each of the databases.
3. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `env\Scripts\activate`
    ```
4. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the main script:
    ```bash
    python main.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [Joan Nieves](mailto:joan.nieves@icimaf.cu).
