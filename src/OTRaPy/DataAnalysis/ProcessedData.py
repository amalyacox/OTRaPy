# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu

import datetime
import json


## IN PROCESS ##
class ProcessedData:
    """
    Class to store processed data for Raman Solver.

    Attributes
    ----------
    database : dict
        Dictionary to store processed data.
    date : str
        Date of the data collection.

    Methods
    ---------
    add_raman_row : None
        Adds a row to the database for Raman data.
    add_tdtr_row : None
        Adds a row to the database for TDTR data.
    save : None
        Saves the database to a json file.
    """

    def __init__(self) -> None:
        self.database = {}
        t = datetime.datetime.today()

    def add_raman_row(
        self,
        sample,
        date,
        path,
        spot_size_path,
        sample_image_path,
        A,
        Aerr,
        pwr,
        dwdp,
        dwdp_err,
        dwdT,
        dwdT_err,
        dTdQ,
        dTdQ_err,
        w0=0,
        w0_err=0,
        l0=0,
        l0_err=0,
        h=3e-9,
        alpha=0.05,
        aniso=True,
        parallel_to_wrinkles=True,
        notes="",
    ):
        """
        Adds a row to the database for Raman data.
        """
        self.date = date
        this_row = {
            "sample": sample,
            "date": date,
            "path": path,
            "spot_size_path": spot_size_path,
            "sample_image_path": sample_image_path,
            "A": A,
            "Aerr": Aerr,
            "pwr": pwr,
            "dwdp": dwdp,
            "dwdp_err": dwdp_err,
            "dwdT": dwdT,
            "dwdT_err": dwdT_err,
            "dTdQ": dTdQ,
            "dTdQ_err": dTdQ_err,
            "w0": w0,
            "w0_err": w0_err,
            "l0": l0,
            "l0_err": l0_err,
            "aniso": aniso,
            "h": h,
            "alpha": alpha,
            "paralell_to_wrinkles": parallel_to_wrinkles,
            "notes": notes,
        }
        self.database[sample + "_" + date] = this_row

    def add_tdtr_row(
        self,
        sample,
        date,
        path,
        pump_spot_path,
        probe_spot_path,
        sample_image_path,
        g,
        gerr,
        w0=0,
        w0_err=0,
        w1=0,
        w1_err=0,
        h=3e-9,
        notes="",
    ):
        """
        Adds a row to the database for TDTR data.
        """
        self.date = date
        this_row = {
            "sample": sample,
            "date": date,
            "path": path,
            "pump_spot_path": pump_spot_path,
            "probe_spot_path": probe_spot_path,
            "sample_image_path": sample_image_path,
            "g": g,
            "gerr": gerr,
            "w0": w0,
            "w0_err": w0_err,
            "w1": w1,
            "w1_err": w1_err,
            "h": h,
            "notes": notes,
        }
        self.database[sample + "_" + date] = this_row

    def save(self, save_path=None):
        """
        Saves the database to a json file.
        """
        if save_path is None:
            save_path = f"/{self.date}/"

        with open(save_path + f"processed_data.json", "w") as f:
            json.dump(self.database, f)
