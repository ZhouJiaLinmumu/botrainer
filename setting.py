import encode

n_numbers = encode.get_encoded_size()
n_hidden = 128
n_categories = 2
n_layers = 1

learning_rate = 0.1
dropout_rate = 0.5

sample_count = 10000

# print_every = 5000
# plot_every = 1000
print_every = 1
plot_every = 1

all_categories = ['non-bot', 'bot']


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
