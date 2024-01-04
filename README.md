# Custom Hugging Face Trainer

## Overview
This repository offers a custom trainer for the Hugging Face Transformers library. It extends the standard `Trainer` class to support auxiliary loss logging, ideal for complex models requiring monitoring of multiple loss components.

## Features
- **Auxiliary Loss Logging**: Enables logging additional loss metrics alongside standard losses, using a custom callback that tracks extra losses within the trainer's control object.

## Installation
Install directly from GitHub:
```bash
pip install git+https://github.com/naba89/custom_hf_trainer.git
```

## Usage

### Logging Additional Training Losss

Use CustomTrainer like the regular trainer, but pass a list of extra loss names for logging:
```python
from custom_hf_trainer import CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    extra_losses=["aux_loss_1", "aux_loss_2"]
)
```
If aux_loss_1 and aux_loss_2 are in the model's output, they will be logged with standard losses.
See [sample_train_script.py](sample_train_script.py)  for more details.

### Logging Additional Evaluation Metrics
To log additional evaluation metrics, utilize the `compute_metrics` function provided to the trainer. Note that `compute_metrics` receives data in tuple format, so you'll need a method to map tuple elements to extra losses. While this functionality isn't directly a part of the custom trainer, you can find an implementation example in [sample_train_script.py](sample_train_script.py).

#### Disclaimer
The implementation provided may not be the most efficient or elegant, but it's designed to work for most scenarios. Suggestions for improvement are welcome.

## Contributing
Contributions to improve functionality or fix issues are welcome. Please submit pull requests or open issues for discussion.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
