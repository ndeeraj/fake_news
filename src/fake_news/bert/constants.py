"""
Some static variables
"""
model_name = 'distilbert-base-cased'
n_classes = 2
data_dir = '../data'
data_fn = ['Fake.csv', 'True.csv']
seed = 2397
test_pct = 0.10

micro_batch_sz = 4
batch_sz = 24
accum_steps = int(batch_sz / micro_batch_sz)
lr = 2e-5
epochs = 25
