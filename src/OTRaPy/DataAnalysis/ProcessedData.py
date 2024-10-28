# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu

import datetime
import json


class ProcessedData:
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
        h=3E-9,
        alpha=0.05,
        aniso=True,
        parallel_to_wrinkles=True,
        notes="",
    ):
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
            "dwdT":dwdT, 
            "dwdT_err":dwdT_err,
            "dTdQ":dTdQ, 
            "dTdQ_err":dTdQ_err, 
            "w0": w0,
            "w0_err": w0_err,
            "l0": l0,
            "l0_err": l0_err,
            "aniso": aniso,
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
        h=3E-9,

        notes="",
    ):
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
            "w1_err": l0_err,
            "h" : h, 
            "notes" : notes
        }
        self.database[sample + "_" + date] = this_row

    def save(self, save_path=None):
        if save_path is None:
            save_path = f"G:/My Drive/Raman_Data/Nanobubbles/{self.date}/"

        with open(save_path + f"processed_data.json", "w") as f:
            json.dump(self.database, f)
