# DNA-Bert
A DNA-bert Model is fine-tuned model to predict whether the sequence has a transcription factor binding sites.
The `demo.py` file contains a demonstration of the DNA-Bert model, designed to accurately predict the presence of transcription factor binding sites (TFBS) within DNA sequences. Leveraging transformer-based architectures, DNA-Bert is capable of learning intricate patterns and dependencies within genomic data, facilitating the identification of regions where transcription factors are likely to bind. By providing reliable predictions, the model enables researchers to prioritize experimental validation efforts and gain insights into the regulatory elements governing gene expression. The `text1.csv` dataset serves as the training and testing data for the model.

# Hyperparameters
1. Optimizer: AdamW optimizer was used with a learning rate of 1e-4.
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

2. Loss Function: Mean Squared Error (MSE) was used as the loss function for regression.
	loss_fn = torch.nn.MSELoss()

3. Batch Size: The batch size used for training and validation data loaders was 64.
	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda batch: collate_fn_regression(batch, tokenizer))
	val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda batch: collate_fn_regression(batch, tokenizer))

4. Number of Epochs: The model was trained for 10 epochs.
	num_epochs = 10

5. Patience for Early Stopping: Early stopping was implemented with a patience of 5 epochs.
	patience = 5
