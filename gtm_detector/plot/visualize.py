from datetime import datetime
from contextlib import closing
import matplotlib.pyplot as plt
import numpy as np

from gtm_detector.data.dto import *


class Visualizer:

    def __init__(self):
        pass


    def visualize(self,
                  in_wells: dict[str, list[InnerResultDto]],
                  out_wells: dict[str, list[ResultDto]],
                  save_path : str):

        out_key ="Результат__16_54_11"
        out_result = {dto.well_id : 
                      (dto.start_date, dto.end_date, dto.gtm_type, dto.reason_stop)
                       for dto in out_wells.get(out_key)}

        dates = [datetime.strptime(str_date, "%d.%m.%Y") for str_date in in_wells.get("date")]
        q_liquid = {dto.well_id : dto.data for dto in in_wells.get("Qж")}
        watercut = {dto.well_id : dto.data for dto in in_wells.get("Обв")}
        q_oil = {dto.well_id : dto.data for dto in in_wells.get("Qн")}
        p_lin = {dto.well_id : dto.data for dto in in_wells.get("Pлин")}
        f = {dto.well_id : dto.data for dto in in_wells.get("Fэцн ТМ")}
        p_zab = {dto.well_id : dto.data for dto in in_wells.get("Рзаб")}

        for well_id, dto in out_result.items():

            try:
                q_liq_data = q_liquid.get(str(well_id))
                watercut_data = watercut.get(str(well_id))
                q_oil_data = q_oil.get(str(well_id))
                p_lin_data = p_lin.get(str(well_id))
                f_data = f.get(str(well_id))
                p_zab_data = p_zab.get(str(well_id))

                start_date, end_date, gtm_type, reason = dto

                with closing(plt.figure(figsize=(12, 6))) as fig:

                    ax = fig.add_subplot(111)
                    
                    ax.plot(dates, q_liq_data, label="Qжид")
                    ax.plot(dates, watercut_data, label="Обв")
                    ax.plot(dates, q_oil_data, label="Qнеф")
                    ax.plot(dates, p_lin_data, label="Pлин")
                    ax.plot(dates, f_data, label="Частота")
                    ax.plot(dates, p_zab_data, label="Pзаб")

                    ax.legend()

                    if not isinstance(start_date, datetime):
                        start_date = datetime.strptime(start_date, "%d.%m.%Y")
                    if not isinstance(end_date, datetime):
                        end_date = datetime.strptime(end_date, "%d.%m.%Y")

                    ax.axvline(start_date, linestyle="--")
                    ax.axvline(end_date, linestyle="--")

                    ax.grid(True, which='both')
                    ax.set_title(f"well_id={well_id} | {gtm_type} | {reason}")
                    ax.set_xlabel("Дата")
                    fig.autofmt_xdate()

                    if save_path:

                        f_gtm_reason_stop_path : str = ""

                        if reason.__contains__("ГТМ"): f_gtm_reason_stop_path = "ГТМ"
                        elif reason.__contains__("обв"): f_gtm_reason_stop_path = "Обв"
                        elif reason.__contains__("жидк"): f_gtm_reason_stop_path = "Жидк"
                        elif reason.__contains__("ИДН"): f_gtm_reason_stop_path = "ИДН"
                        elif reason.__contains__("МУН"): f_gtm_reason_stop_path = "МУН"
                        elif reason.__contains__("остан"): f_gtm_reason_stop_path = "Останов"
                        elif reason.__contains__("Отказ ЭЦН"): f_gtm_reason_stop_path = "Отказ ЭЦН"
                        elif reason.__contains__("Смена ЭЦН"): f_gtm_reason_stop_path = "Смена ЭЦН"
                        elif reason.__contains__("ТРС"): f_gtm_reason_stop_path = "ТРС"
                        else: print(f"{reason}, {well_id}")

                        fig.savefig(f"{save_path}/{f_gtm_reason_stop_path}/well_id_{well_id}.png")
                
            except Exception:
                continue

            finally:
                plt.close(fig)

