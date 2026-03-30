from main_designed import main as main_designed
from main_pendulum import main as main_pendulum
import os

if __name__ == '__main__':

    # delete existing CSV file to avoid appending to old results
    csv_file = "qp_results.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Deleted existing {csv_file} to start fresh.")

    print("Running designed cases...")
    main_designed()

    print("\n\nRunning pendulum cases...")
    main_pendulum()

    from utilis.plot_stuff import plot_charts, plot_reg_counter
    plot_charts()
    plot_reg_counter()