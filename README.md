
# Generative_Nazam

This project focuses on generating **Nazam** (a form of Urdu poetry) using a Deep Learning model. The model, built using **GRU** (Gated Recurrent Unit), is trained on a dataset of Roman Urdu poetry to generate the next part of a **Nazam** based on the user's input.

## Features
- **Generative Model**: Uses a GRU model to generate the next part of a **Nazam** based on a user's input.
- **Roman Urdu Text**: The model is trained specifically on Roman Urdu poetry, enabling it to generate authentic poetic text.
- **Web Interface**: The project also includes a Streamlit web interface where users can input starting text to generate poetry, view generation history, and download the results.

## Project Structure
1. **main.ipynb**: Jupyter notebook for training the GRU model on the dataset and saving the trained model and tokenizer.
2. **roman urdu poetry.csv**: Dataset of Roman Urdu poetry used for training the model.
3. **poetry_gru_model.h5**: The trained model, ready for use.
4. **training_history.pkl**: Stores the training history (loss, accuracy) of the model.
5. **word_encoder.pkl**: The tokenizer used to convert text into tokenized sequences for model input.
6. **app.py**: The Streamlit app that loads the trained model, generates poetry based on user input, and displays the results.
7. **poetry_history.json**: A JSON file storing the history of generated poetry for the app.

## Requirements

Install the necessary Python libraries using pip:

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` for the web interface.
- `tensorflow` for the deep learning model.
- `numpy`, `pandas`, and `scikit-learn` for data handling and processing.
- `keras` for deep learning functionality.

## Setup

1. **Prepare Dataset**: Ensure you have the dataset file `roman urdu poetry.csv`, which contains two columns: `Poet` and `Poetry` (Roman Urdu text). The `Poetry` column should contain the text of the Nazams.
   
2. **Training the Model**: 
   - Open and run the `main.ipynb` notebook to preprocess the data, train the GRU model, and save the model (`poetry_gru_model.h5`) and tokenizer (`word_encoder.pkl`).
   - The model and tokenizer are saved for future use.

3. **Running the App**: 
   - After installing the dependencies, run the Streamlit app using the following command:
     ```bash
     streamlit run app.py
     ```
   - The app will allow you to input Roman Urdu text, generate poetry, and view/download the generated results.

4. **Using the Model**: The trained model can be loaded for further inference or deployment as needed.

## Notes
- Ensure that the appropriate version of TensorFlow is installed for compatibility with your environment.
- The app can store up to 50 generated poetry entries in `poetry_history.json`.
