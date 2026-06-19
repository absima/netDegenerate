"""
Run the full simulation parameter grid in parallel.

The parameter tuples come from simulation_pipeline.list_paramList and follow:
    (ilb, cp_index, inet, idtyp, idxprun, istage, iscale)
"""

from simulation_pipeline import *


def main():
    """
    Execute parent and degenerated simulation batches.

    Returns:
        None
    """
    start_time = time.strftime("%H:%M:%S", time.localtime())

    # Single test point:
    # param = (1, 0, 0, 0, 0, 0, 0)
    # simulateAndStore(param)

    for ips in range(2):
        paramList = list_paramList[ips]
        if ips == 0:
            arr = np.array(list_paramList[ips])
            sorted_arr = arr[arr[:, 3].argsort()]
            cutt = np.where(sorted_arr[:, 3] == 1)[0][0]
            paramList = [tuple(j) for j in sorted_arr[:cutt]]

        Parallel(n_jobs=-1)(delayed(simulateAndStore)(param) for param in paramList)

    finish_time = time.strftime("%H:%M:%S", time.localtime())
    print("started at: ", start_time)
    print("stopped at: ", finish_time)


if __name__ == "__main__":
    main()
