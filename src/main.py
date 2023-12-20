# Project
import app.core.chapter_02.excercise_05 as C02E03
import app.core.chapter_02.section_03 as C02S03
import app.core.chapter_04.example_02 as C04X02
import app.core.chapter_04.excercise_07 as C04S07
import app.core.chapter_04.section_03 as C04S03
import app.core.chapter_08.example_01 as C08X01


codes = {
    "C02E03": C02E03,
    "C02S03": C02S03,
    "C04S03": C04S03,
    "C04X02": C04X02,
    "C04S07": C04S07,
    "C08X01": C08X01,
}

codes.get("C08X01").play()
