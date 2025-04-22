import sys
import os
import pytest
import logging
from pathlib import Path
print("system path prints here>>>>>>>>>>>>>" ,sys.path)
# add the parent directory to the system path
# to import the bikeshare_model module
# This is necessary to run the tests from the root directory of the project
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
print("this is a message>>>>" , root)
import pytest
from sklearn.model_selection import train_test_split

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name = config.app_config_.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    )

    return X_test, y_test