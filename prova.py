from multiprocessing import Process, Manager
import copy 

def worker(shared_list):
    # Access the shared list and print its value
    sc = copy.deepcopy(list(shared_list))
    print("Worker:", shared_list)

if __name__ == "__main__":
    # Create a manager
    with Manager() as manager:
        # Create a shared list
        shared_list = manager.list([1, 2, 3, 4, 5])

        # Start a process and pass the shared list
        p = Process(target=worker, args=(shared_list,))
        p.start()
        p.join()  # Wait for the process to finish
