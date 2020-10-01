import pathlib as path
from ann import ann
from pickle import dump
from mnist_loader import load_data_wrapper
train_d, valid_d, test_d = load_data_wrapper()


dir_name = "pck_nets"
p = path.Path(dir_name)
assert p.is_dir()

nets = []
dummy_sizes = [10, 10]
for json in p.iterdir():
    if json.is_file() and json.name.endswith(".json"):
        net = ann(dummy_sizes)
        net.load(json)
        # nets.append(net)
        name = json.name.replace(".json", ".pkl")
        with open(path.Path.joinpath(p, name), "wb") as my_path:
            dump(net, my_path)

# print(nets)
# best_3 = ""
# best_3_score = 0
# best_4 = ""
# best_4_score = 0
# best_5 = ""
# best_5_score = 0
# for net in nets:
#     accuracy = net.accuracy(valid_d)
#     if accuracy > 9600:
#         print(net.sizes)
#         print(accuracy)
#         if net.sizes == [784,85,75,75,10]:
#             print(f'cost: {net.total_cost(valid_d, 0.1)}')
#
#     if len(net.sizes) == 3:
#         if accuracy > best_3_score:
#             best_3_score = accuracy
#             best_3 = str(net.sizes)
#     if len(net.sizes) == 4:
#         if accuracy > best_4_score:
#             best_4_score = accuracy
#             best_4 = str(net.sizes)
#     if len(net.sizes) == 5:
#         if accuracy > best_5_score:
#             best_5_score = accuracy
#             best_5 = str(net.sizes)
#
# print(f"best 3: {best_3} of score {best_3_score}")
# print(f"best 4: {best_4} of score {best_4_score}")
# print(f"best 5: {best_5} of score {best_5_score}")
