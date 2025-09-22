cleaning_rules = {
    "HHADULT": {"remove": [77, 99]},
    "PHYSHLTH": {"replace": {88: 0}, "remove": [77, 99]},
    "MENTHLTH": {"replace": {88: 0}, "remove": [77, 99]},
    "POORHLTH": {"replace": {88: 0}, "remove": [77, 99]},
    "ALCDAY5": {
        "replace": {888: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) / 7 * 30",
            "201-299": "lambda v: (v - 200)",
        }
    },
    "AVEDRNK2": {"remove": [77, 99]},
    "DRNK3GE5": {"replace": {88: 0}, "remove": [77, 99]},
    "MAXDRNKS": {"remove": [77, 99]},
    "FRUITJU1": {
        "replace": {300: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 30",
            "201-299": "lambda v: (v - 200) / 7 * 30",
            "301-399": "lambda v: (v - 300)",
        }
    },
    "FRUIT1": {
        "replace": {300: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 30",
            "201-299": "lambda v: (v - 200) / 7 * 30",
            "301-399": "lambda v: (v - 300)",
        }
    },
    "FVBEANS": {
        "replace": {300: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 30",
            "201-299": "lambda v: (v - 200) / 7 * 30",
            "301-399": "lambda v: (v - 300)",
        }
    },
    "FVGREEN": {
        "replace": {300: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 30",
            "201-299": "lambda v: (v - 200) / 7 * 30",
            "301-399": "lambda v: (v - 300)",
        }
    },
    "FVORANG": {
        "replace": {300: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 30",
            "201-299": "lambda v: (v - 200) / 7 * 30",
            "301-399": "lambda v: (v - 300)",
        }
    },
    "VEGETAB1": {
        "replace": {300: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 30",
            "201-299": "lambda v: (v - 200) / 7 * 30",
            "301-399": "lambda v: (v - 300)",
        }
    },
    "EXEROFT1": {
        "replace": {888: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) / 7 * 30",
            "201-299": "lambda v: (v - 200)",
        }
    },
    "EXERHMM1": {"remove": [777, 999]},
    "EXEROFT2": {
        "replace": {888: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) / 7 * 30",
            "201-299": "lambda v: (v - 200)",
        }
    },
    "EXERHMM2": {"remove": [777, 999]},
    "STRENGTH": {"replace": {888: 0}, "remove": [777, 999]},
    "FLSHTMY2": {"remove": [777777, 999999]},
    "HIVTSTD3": {"remove": [777777, 999999]},
    "BLDSUGAR": {
        "replace": {888: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 365",
            "201-299": "lambda v: (v - 200) * 51",
            "301-399": "lambda v: (v - 300) * 12",
            "401-499": "lambda v: (v - 400)",
        }
    },
    "FEETCHK2": {
        "replace": {888: 0, 555: 0},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 365",
            "201-299": "lambda v: (v - 200) * 51",
            "301-399": "lambda v: (v - 300) * 12",
            "401-499": "lambda v: (v - 400)",
        }
    },
    "DOCTDIAB": {"replace": {88: 0}, "remove": [77, 99]},
    "CHKHEMO3": {"replace": {88: 0}, "remove": [77, 98, 99]},
    "LONGWTCH": {
        "replace": {888: 0, 555: 50},
        "remove": [777, 999],
        "ranges": {
            "101-199": "lambda v: (v - 100) * 365",
            "201-299": "lambda v: (v - 200) * 51",
            "301-399": "lambda v: (v - 300) * 12",
            "401-499": "lambda v: (v - 400)",
        }
    },
    "ASTHMAGE": {"replace": {97: 5}, "remove": [98, 99]},
    "ASERVIST": {"replace": {88: 0}, "remove": [98]},
    "ASRCHKUP": {"replace": {88: 0}, "remove": [77, 99]},
    "ASACTLIM": {"replace": {888: 0}, "remove": [777, 999]},
    "SCNTWRK1": {"replace": {98: 0}, "remove": [97, 99]},
    "ADPLEASR": {"replace": {88: 0}, "remove": [77, 99]},
    "ADDOWN": {"replace": {88: 0}, "remove": [77, 99]},
    "ADSLEEP": {"replace": {88: 0}, "remove": [77, 99]},
    "ADENERGY": {"replace": {88: 0}, "remove": [77, 99]},
    "ADEAT1": {"replace": {88: 0}, "remove": [77, 99]},
    "ADFAIL": {"replace": {88: 0}, "remove": [77, 99]},
    "ADTHINK": {"replace": {88: 0}, "remove": [77, 99]},
    "ADMOVE": {"replace": {88: 0}, "remove": [77, 99]},
    "DROCDY3_": {"remove": [900]},
    "_DRNKWEK": {"remove": [99900]},
    "MAXVO2_": {"remove": [99900]},
    "FC60_": {"remove": [99900]},
    "PAFREQ1_": {"remove": [99000]},
    "PAFREQ2_": {"remove": [99000]}
}
