from random import uniform, randint


if __name__ == '__main__':
    mi_range = (50, 500)
    lambda_range = (20, 100)
    h_range = (5, 15)
    k_range = (2, 7)
    mutation_range = (0.0005, 0.05)

    mi = randint(*mi_range)
    lambda_ = randint(*lambda_range)
    h = randint(*h_range)
    k = randint(*k_range)
    mutation = uniform(*mutation_range)

    print(f'{mi}, {lambda_}, {h}, {k}, {mutation:.4f}')
