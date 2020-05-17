from edt import EDT


def read_seeds():
    path = 'datasets/seeds/seeds_dataset.txt'
    x = []
    y = []

    with open(path, 'r') as file:
        for row in file.readlines():
            vals = row.split()
            x.append([float(tmp) for tmp in vals[:7]])
            y.append(vals[7])

    return x, y


if __name__ == '__main__':
    x, y = read_seeds()
    tree = EDT()
    tree.fit(x, y)
    print(f'Accuracy: {100 * (1 - tree.eval(x, y)):.2f}%')
