# Project Title

## Requirements
1. The Python dependencies for this project are listed in the `requirements.txt` file. Install them with:
   ```bash
   pip install -r requirements.txt
2. Download the training set from the following link: Training Set Download Link(https://drive.google.com/file/d/1DYSztWyWMi2pSbQeElF5k9W-FbDJzQ8O/view?usp=sharing). The testdata folder already contains 5 test data files.
3. Use the file pnn.py to run the code. Make sure to update the paths for both the training and test data within the script:
   if __name__ == '__main__':
    # Update paths for training and test data here if needed
4. Each time an activity period is completed, a new thread will be created. You can add custom code in the action_task function, as shown below:
   def action_task(label):
    # Add your custom code here
5. Just replace `URL_TO_TRAINING_SET` with the actual URL for the training dataset download link, and your `README.md` will be ready to use!
