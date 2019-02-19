import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
META_ROOT = os.path.join(PROJECT_ROOT, 'split')
TRAIN_META = os.path.join(META_ROOT, 'train_meta.txt')
VALID_META = os.path.join(META_ROOT, 'valid_meta.txt')
TEST_META = os.path.join(META_ROOT, 'test_meta.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
TRAINING_OUTPUT = os.path.join(OUTPUT_DIR, 'train_output.txt')
EVALUATION_OUTPUT = os.path.join(OUTPUT_DIR, 'eval_output.txt')
CHECKPOINT_DIR=os.path.join(PROJECT_ROOT, 'checkpoint')