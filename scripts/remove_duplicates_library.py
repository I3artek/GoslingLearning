import difPy, json

def main():
    path = '../ageSplit/020'
    dif = difPy.build(path, show_progress=True)
    search = difPy.search(dif, similarity='similar', show_progress=True)
    search.delete(silent_del=True)

if __name__ == '__main__':
    main()