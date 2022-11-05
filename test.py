a = []


def func():
    a.append(1)
    a.append(2)


if __name__ == '__main__':
    func()
    for x in a:
        print(x)
