<div align="center">

# RPX

### Same Scene, Different Story:<br>Evaluating Robot Perception Across Scene Phases in the Wild

<a href="https://jishnujayakumar.github.io">Jishnu Jaykumar P</a><sup>1&ast;</sup> &nbsp;·&nbsp;
<a href="https://github.com/Optimax14">Itay Kadosh</a><sup>2&ast;</sup> &nbsp;·&nbsp;
<a href="https://narendhiranv04.github.io">Narendhiran Vijayakumar</a><sup>3&ast;</sup> &nbsp;·&nbsp;
<a href="https://github.com/srinandithak">Srinanditha Kamath</a><sup>4</sup> &nbsp;·&nbsp;
<a href="https://saihaneeshallu.github.io/">Sai Haneesh Allu</a><sup>1</sup>
<br>
<a href="https://github.com/GovindRangappa">Govind Tyagi Rangappa</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://github.com/AnimeshMaheshwari22">Animesh Maheshwari</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://jwroboticsvision.github.io/">Jikai Wang</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://yuxng.github.io/">Yu Xiang</a><sup>1</sup>

<sup>1</sup>UT Dallas &nbsp;&nbsp; <sup>2</sup>UF &nbsp;&nbsp; <sup>3</sup>IIIT Hyderabad &nbsp;&nbsp; <sup>4</sup>UT Austin
<br>
<sub><sup>&ast;</sup>Equal contribution</sub>

<br>

<a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img alt="Dataset on Hugging Face" src="https://img.shields.io/badge/dataset-IRVLUTD%2FRPX-ffcc4d?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAADwAAAA4CAYAAAChbZtkAAAABmJLR0QA%2FwD%2FAP%2BgvaeTAAAOc0lEQVRoge2aeWxcx3nAfzPv7fl2uSRFUhQlkhJFiRQt0bTko47jKJZdNy2c1HBrJ3VrOPKJwK2BoDUKJAastsjR9o%2BirRPEjSUhTtMmaZwi9YEGTWPBrpQoli1ZpyXSkkiJpHgfex%2FvTf%2BYt9wld3lKcYq2H0Dw7cw333zfzHfNNwP%2Fx0B82BOqvTVh0lkTAEM64smJqQ9z%2Fl%2BawGoPkqbATSh5B6gbQVwHrAf8c1DTQB9KnAbnCBhvcDl6WOwh98vg65oLrPZb23DU4yAeAFavkMwYQvwA5bwoHk0cuZb8XTOB1b7AR1DyOeDuMt1jeEQvPjOGVzhIqed1HEXGkaSdIDmnCUVdKWHeQvLn4pH4T64Fn1ctsPpWYC22%2FBsU98%2FqMMQJav3j1HibCJgblkQsZV9iNHOBsWQFGdUJyMJEvI5pPi0%2BO%2FXB1fB7VQKrvdbvA18DIm5TlqDnMBusBoJGy9XQJmn30Ru%2FSDR7MwW7T4D6Y%2FFo4hsrJbsigdULeDCCf4sQn5tp9BpvszlcR8BoXikzZSHtDHBuqpeUc2tR63eQ8cfEblLLJbdsgdULBDGtl4FPuBSmWWedpN7%2FkeXSWhaMpN%2FmQmwDghrNCG9hez%2B53LC2LIHVfvw41qvAnQAYooeOiBe%2F0bQcOiuGtDPIqckJbNXhthyGwK%2BLR0ejSyUhF0fRoPYgcYLfJi%2BsKU%2FSWVX7oQkL4JNr6KpqxivfcVtugeQP1B7MpZJYssA0Br8I4nf1KHGazkgzpogsMuragxQWnZUdmOKY23I3jdZfLnX4klRa7Q1%2FFJwDgIGgn64qL6asXQG71w5sNc3RiVGUagEUUt0jdideX2zYojus%2Fg4fOPsAA8jSXjHxKxcWwBAVdEQAEoDAEf%2Bg%2FrG6YrFhi6u0ZX0e2ARAtfcgIc%2FWq%2BP0GkLQaKE2kE8915JOP7vYkAVVWn09UoUvdwGIIOhje3UdUsxN%2Fn%2FVYPPueDe2agdSoDaKRxMD8yEv7N182c%2BB65iarMtIsaBHHpsyef6HtRx536LCsvn0rnE%2B9dGlh8mBUQ%2FPv1zHez1%2B6qpsHvqNMXbtWDTiGDSH4pyPAvgR4vPAM%2FMhz7vD6vsYRK2LwDqglxtXNSLmN4FsTvDZL6%2Fn4qB3VvszDw5x7%2B2TizFNLCl56C%2FWMzzhKTAn4EtP9LOzK7boeN4dP4OttiCYQMQb5svC5rfhmHUHWlio819cSFiAX5yxSoQF%2BO5PqhZnFvjpO%2BFZwgIoBd%2F7z6WNpz4wpgdRhR365Hxo8wvhqPvcL5uGwHUl%2FWNp6I3DdBaAyahRlsxUvHz7XJiMlbeuqXhRe86BywnoT0JOzUZc7e8CkgCIGd5LYKFd0%2BdaQ5zBI2tm9Yyk4HwMhlNwdhriOTo3JZFCMxEKOHx61wRtTSl2tCUWmKIA2zfHiYRsfuvWKW7YnJxp79pUNP5sFAaTMJCAc9OzCRgihM84kedd7SkvW9llVS8E1yDERgAqPKMlCNFcye%2FG%2BgxP3TfK1%2F%2B1hmcfHuT265dgd0WwtSXF63%2FdA0AmK3j8r%2FSh64lPudPbChJF88Zz4CiQRW6o0kwxZANU0xRqh9jpufOU1yOTrpnviNcq6Y94tEqD9iwV2vY%2Bc9c499w2SSjgLFXOsuD1KL7xTB%2BmVHhMV3UNASETYq7QYXO2sACV%2FiqGXL6UugFYosBCbkKpPOH6kv5VPs1ALAdVXggW7PRqhc1DwFuGzuYKbU4KqCuTDgSLDjJKbC5Hd544rBryH%2FhkqcAAlV7992GCIaA%2BMH%2B%2FKSIIFUcJC2gohzKP0xKVACiSCOEpj%2FM%2FFYQO%2BsKpLNc7j8DK4%2FamVzLlpe4Az3%2B1kXO%2FKDX%2FhUA5cOiVap7%2FSiPZzArLbYKM%2Fi985brLq7QSOW0oakU6%2B97hMP%2FcG%2BS7%2B4Pc%2Br0sD31inM67JufN6%2By05I1%2FqeHbP4%2FQY%2Bs9%2BO0LPhrbll2yAoWrkSpbrnsep6WmtbwiiFI5hFhyRQGgqirnzg2HEh4O%2FXA1rT%2Bq4a6WJNu2xalbm8a2Bf0XAhw5bvEfA35G1ezVqF69wosHRVj%2FF2WT%2BHl2mEH3S5B2hvAba5czZ%2Fv2GOLVWopzoR7boKc7BN2hRcc3GQ5W5QoEtokzUzIWg%2BVQ5rPhnpnPmD1EzoHRFAyldMBfBFatybDNt%2FKroZ3NycWRALIOjKYLfCWyfYVO1V1uSPkdls5xHDe2jqYkfTmd6eShwgMtIfC46zWdhbQNAVMnB8BnPj7J8R%2FPzkjXKMUO2%2BFVU9P%2BmG3TKyW9oqDOPuC%2BB0YKgzKOpi%2FQCY%2FpzjmQ1GmmU8SXzyh4ZqXeW7LAYnfqotprXQbWEc2WHhyms3BmClor4EJsdsoX8UJriJ33jnHrwQg%2Fi2kfcr%2Bd46lsDo%2BCHDAoBV%2FK5ADFPo%2FJflOz8sh109Q1u8FhJAW9CWaSIClgQ0jv5pUyWpC217hfUS4nTpQiLHQe3hf6Dsp5cAZlcxjCXhhK6hNLfrQqM7jGBxtCRCc8%2FNFzzWxOwRcyBaeZBmwEwaLBX%2FOYjNfbfPHPepFS6Xz97FQp%2FeI5A4bOvkwBxybBnsnOXiMe%2Fx3xdGlYLRFYvWj9CYI%2FBWbrY1c1eFz0K0m4VHSKWReE1X64GC%2Fk2FsrIWCQiBo4L1cSejME82WdPofs%2FVMYd0SRhivNmSmduhoStkX02OMTRWOkniOfT5%2BYzJCy82FUATaI15HqSbE7fqWswOrF0B8i1N%2BXZWpjCKqLYvn70xB1d21HtZ7YUfDOuG6r90NjUeIxbsIhCy559LcBrMpBWwpuikOgaCuTNpx0qyQtIZ27gz6S5he0q6rgQ4rnLYWjXIrfnL9gn7FhpRDsU18AwBTDNFonGct0MJ3RufQHMe2s8k5jUxiOjeuVzzsdKaDWr21vJKXz3jxT1Tm4Z4n1rYEi7akqWmSvS2t9kcMEODedAfTueo0YDf7DTOcqGE%2FfBNzAuuBvQuIVKA5L%2B2pCgDb6Jus4Nb5dtIXr2RJRBFyPfWKy4K0NARt0jCdbpKtr3eTeRlcmlguxLEzo7JB1wdmBczwNfgNqixahLw7RnBcpYF1QcX2lh1r%2FnbRYhUOPKJycCuQeGY0Bui40kGxFMQRAyBR0RLSN5pRWtbzQ1V7wGVrV8hrpkRoX9C5Plc3wyoNS2g8oQBbRAb0IaQdaw4W2ywkdgwMGdERgTUCgI5tiOFW4OHfEhVKBNZwEIGWv59iETdY5rLEENFmwMeTgKMXpyYKAm0LalruLPGqjVVD98zEdo5cCF%2BLafgFaQwWHNJ6Gnqi25by2DaZ0HF7lzbElUmhXapTe2M%2FoS3w8v4zAmVKB9wZ3AztnfuecBo5N3Ex%2F4i0cpY2v2idpj2SRUnFqSu9IwNSMTOVg3FVFAbRX5Onould2kcJAX5GHr%2FXpJAO0j%2Fggpmk2u05wJA2X49BoxWkJmxhCCxbP%2FRfHJiTD6eK7aoFkb%2BFHfhn2Wj8G7sYUo3REusmSZiBeyVS2C1NMsj6UoMqrD9VZlaUvZhIyBasDeg3Ttg4VRVkTExm9M6BLMm2R8pF%2FJKVVGSDkgS1zrohStjYVQ%2BgFPDkF660olV6t3xk7Rm%2FiIpOZrXjlMOuCJwh7LK6kbIaSt2kB7fXisVRvIdMSogKlwGuM4jNuwYdkcwVk1bt0TwfpibYT9sTZYFn4DA8bwwUVFmhnMheqvNqJ9Sd1IjGSKi3N5JxCTPdKaAuX0immbUrorLSRQiMOJmP0J%2F0ImmkNv0GV9xbyd9iVngMMuY7TMUMw65WMOghAItdOf%2BINFFqNPWI7HZHNtEfeIpXLcGbKJuUUBF0MGoJgues6WOZ8O5wqOMH2SGlhrhxIoVfgfDTN5YRFvf8wO6oTVHnvAIIApOxDdEe3uyNGUbFzUJxLC%2FH2TM46kLyT4dQYjdZPWeVrR9BA2LydrupJxtJvk7OvB7lAcWkONFk6c8rYWj2Ld8wt5FPh0SaxHPAZp9le5cGQt7ktGdL2ES7Gw0xni%2B34mHiSLMy24X8Cfg8p42yrOEpOwUAixGR2G7X%2BY6wNGJiyi5VAzoGjblrYVjFT1gXg%2BKS2%2F8bgwgW6hUAxzHi6m0uJLQiyNFmnCHvC9MZxkw%2BHVDAinhopvt9QHhAgVRav0YqXelorIOsc5Xw8xNGJNjZYb1Lj%2F9jyBS76NuaobH6z0yss79rOOY5PrAWxjQ2ho1R6bwR2AeCVb7pYAjNnwKw4LF7RzKlKTk3FcBx9vPLIG2gLt9FVfaJsUX4pMOLarhSFeJkHy93t0RQoVe7stTAYMkx75Dhd1QaV3p2AhWCaK8kDXEm6m6MO5p83FVR6D5JG60fAPegehzr%2Fz2kIhDBl57IZycN0RnE2qufxyivU%2BHswXadjK5toLsR0RptKpVexKbzy14GKYSbSp%2BmNd5JT1a4cE%2BTE7eKJ2Cn9sxj%2F%2B3iZtr6K4GkKygYBo5t1gUtEvK2IhS%2FFZ0HC1pdeiyUdxdAQgLXBpeMrEqTt9xhMCcZSOwpVSwCOIuWDYnf0%2FXxD2dVU37Q6MdRzKHEvc9PPoHmeau8lIh6B31iFFA1A4RJXESWnzjGcTDCY%2BjWUyjOQBcoX9RVpBAb5qBH2nKA5NIFPNiMpfsqYQal%2BsmqYaDbBRCbCVLYDR829d%2BlG8BVC8ZfEA8zKaxd%2B47HP3wLmbpT6A%2FTj7vkgjSmSOBg4am7moFDiy%2BKx2LPqm9ZqTOrI5evdIoujRngicYX9gZtR8t%2Bg5AlxBlMksDFRaqGSZxJ4Bam%2BRW%2Fi38We8uWGpb3TUgiXoXuBu4AuFnsfokcexJHPisdjB5Y0z0uhOjLOHoR4mHwCsTBcRnEAxGsI%2F2tLeYK40te0QYzAVpBbkKoZRQ1IfUhVagw4i7LfFI%2BnLixMaR76L2FhB3fi0AWsBZlf3FFdM1c9eJwT4uFk%2F0ro%2Fz%2F8b4b%2FBoZuUT9l1ZE1AAAAAElFTkSuQmCC"></a>
<a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img alt="Toolkit docs" src="https://img.shields.io/badge/docs-toolkit-0d9488?logo=materialformkdocs&logoColor=526CFE"></a>
<a href="https://pypi.org/project/rpx-benchmark/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/rpx-benchmark?label=pypi&color=4F46E5&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAEAAAAA4CAYAAABNGP5yAAAABmJLR0QA%2FwD%2FAP%2BgvaeTAAANCklEQVRogdVbe2xb13n%2F3fflvbx8SRRFUhL1siVRT0uKEntJCqdO0hjJ1m5os3VttyLLgHUPbNkL2Zoiw4YGK7ANWxu7K7DWLZYVdd06cVA4S%2B2otmvXD9lyJCuyLcl6i3pQpCg%2BRPJe3rM%2FVLmxLImXFJVkP0CAoHO%2B73znd873ne8754rC%2FxMMDQ1Zjp269m935lae0BimpL7CNV5TKh%2F9rSf2vUxRlJavXqqQRu4EDn33jRduToWfm0tou2%2FPJ1idkLttHEOjvdqVbqwoGqwvs37ryUc6D%2BWq%2FyNJwLf%2B%2B8dP35mN%2F%2B18TG0fmItLSVXPKqOYODy02xOvLbX1%2Bmzii089%2FtDPjYz1kSHgx2%2B%2B1XptKPRVQZaeLHGY6SNnblGJZO47u9RmwoHWat1dpPz0cwcf%2FES2%2Fmxe1hYIQ0NDljfeuf7yyGzsU98%2BM1YxuZSiAYCmKLRWOtFSUQRdV%2FHa%2BWHoW2wCkWfxSGMZBNGM0cUVHOtbpD%2B%2F12wxYsOHQsCrR47%2F5VAg8nsvHTlTf2s%2Bwb3frwFAJwS9o%2FPoHZ2HwDL4mL8Mu9w2DE%2BH8c5703f7PewvQ4nDhoWYhvPjS9BJMmdbPlAC%2FuM7x6%2F2jIYbjlycNKU1kl0AQErLoPvGFLpvTKHEIuH5A02QTVymfzbFXBuPoCcQ3JZN9Lakc8QvhoL%2BeDpjanGZ0V5hA21wdKfVhKcf2I32hiq8dTsKFTx9fjiEFTWzqYyZWW4dvn78y4SQLUf5QHeAToDhYALA6hHms%2FKwChxiOoXh%2Bdg9fdf79anh5ZzGUpiAVJX6h3%2BMXDvx0nTvvwzGmd1f293yzP%2Bs7%2FehBUE1o2M0nAaQhkPm0VIqQxRYiGYJTrsTczE1b79eAw0VVu0Ub8WpVhWO10LX%2Fvhwkm08v0SavuxvffQa8AEScO7cpd2H3urbcLxQPI1QPA0AeLzNjp8MzG%2BtjMr99OYQgl09YYF64ikrVf%2BJwPVXAlGq6bUdJeDo0aO8w%2BF%2BRZLk36yoKKlg6BvZvZ7K3oWQfPMXFkvsE9C4CkrGrAe048COEHD69PnnWZb9w9JSR0t5eSlP5bFiXoeMl57tgtdhxpf%2BsxuTwVh2oU0QZfdC5RpAIwUrroLCEABghfIXzgXOnLn0mK6Tv7HbLXurqz0WjstP9RpVX%2F38PtR57QCAhjJHzgSkYSdB4TmKZXSY0Q%2BFnNuwX8EIkGX%2BpK6DLypSkO%2FkAYCiGCgm7u7kVU3Hu2PGzvpimUeb14pSqwBR1FFMnwWypBsFI8BkEuFxFyGd1nBzcAQZXUdFhQeKIuekhxAgllTRP76IIkXEv57oxdxSYtP%2BPEOjxWtBTbGMaCqFk%2B8tgBDgz%2Fa7CQzUOgWPATzPorLSDQCIRhOYGA%2BAYWjU1PqM2ANglYTnXz29aTsFoNGtYJdTgcARHH93Fj0TS3nZWzgCyP17TVEkKIqEcCSOb%2F%2FoZ%2BioLYXVouDKnVmkDJS461HrlOB3KSiRacxFUxgIRLdt9rYI6O7uFilK%2FGdRFJ4BwL%2B%2FLa1qOHvlNi7fnMH16Qjmo6vnPE1ReKazChLHYWR%2BCT0j81hfDL0fDolDq9cKX5EJ16ciOHFjDk57JQnG0ltuJ4raOgVeQ14EdHef%2FRzDiH%2FqdNpafD63SFEURu9MAgCu9I%2FiQt8EBmeXcWs%2Bdt%2FG0AnBG1fuAAAsJg7P7tuFVFpH38QChuciAFb9utljQa1zza%2BDIEM5GkmMuZthAs6fv%2FyopmW%2BYrNZunw%2BtyII9yw4fnFjDOd%2BeBH9gRi0jLHtvbyi4vvnbwMA6jxWPP%2FxJvAchxK7jNffncPVyfz8OhcYIuDixZ5gSYnD4XaXbErrqb4ZxJJptJXK4AQOl8ZCW15irKG6REFrjRcpsHhjMILPdHpw7NL41kJ5Z4KrSBMLglon4nqp0xABdpvFJoo8NTAwDIah4fN5YDKJ62wimIikMRFJg2No1BVLsAgsQmkdQ3P3JjEmnsb%2BlipwogmDM3GcvJnbShu7SVgvwyKotiKm14Fjk%2FByr2OBcS4YdgGTyKOm2gMACIeXMDy8DEkSUVVVBpqm7%2FE5NaNjcH717LaZODS7JMgiB9lqhlmxYzKUwjsjUQDxPKaSGyKZXVjKNEFnrCjnj8FJrt7TbogAsi6S2e0W2O0WEEIwPjaD5eU4QG28LksrKpZWVADA43uc%2BN%2F3ts7q8igb7oPXJkDiGTKqfYFysT%2BDjzn%2By4nc39cQARsVMxldx%2FRUEGlVh6%2FSA4q6vi2j78Jg9F4PWWDRWW5Bm4fgUc9Z0KZa4ta%2BV%2FhMcHYuhGg0heJiK3bXVeZlbKHAMhT8LjM6ygQ8XHYHfvkwKKxG3gBqjekw0mklmcLIyAxMkgifzw2GYfKzuADbm6IAgQE%2B2%2BlEl3sRD9gOg9lGLDFEgK5nGJ0AsizkP3kA22HApfDY41XwQHkSfvcUqSKvbp0JEmPRxBABkmSCq8QOTdNw8%2BYoNE1DeXkprFbFiPivkOP5td6vvdwFAECA%2Fn1g8wvhtbEKR8DdziyLSp8LABCLJTBwYw4MQ6N2l68g0RsAQAEtHmVDv94JGCTg%2FqUzmyWYzRIAYDYwj311HhRbY7g8Mo8VNbc3PYoC%2FC4z6l0KPGaQb378VYrGSk468oFOaHFTAnp6eqxLS8l%2FN5vlA1SW3VRcbMXPB2cwFExgf70LiiRhKhRDz52tKz2XwqPZbYXXJuDCaBg%2FejeAutKirJMn2wgmmi4ipHdghW1FkqmX7yPg9OkLf87z7HN2e3GD31%2FEUBSFifHpjXTdrzyj46cDAQCA2cTg19vKoIPD7bkwbs6EAaz6dUeZBWUOE0YX4zh1eyH3WeScC1MIa42IUq2A0oXy6kZQFIX5%2BcVFFgC6u7ufpijTC0VF1q6qKq%2FMMLm%2FmK1fkthKBq%2F3rpbIDW4Fv7t3FyxWBXaLhDf750BGch4iZ8QzZQiRB6CJbfDWdcHO8%2Ff1YS9d6omUljqVkpKibYWxrRZlMBDFYCCKx%2FfU4tzY1re7%2BVyhvx9JUoxFtQMxtoTI1V%2Bnyi1bv5KzzmK7meMY6ubgCHQdqKzyQJJM2zJiO8in0ssQHiGtDXFUQ2TC8PJvI8DXEMViMZYKm0T%2B7kXm8nIMIyNTEAQOtbUVq5XeRxIUQqofMbIbYESUccfhxOWctbDrKz2LRYbFIoMQgsmJGUSjKzCbTSBk%2B5VaIbiM6jVIZBSM6Z%2BFhzsJBwa2pY%2FdzOcoioLL5YDLtVoATUzOgWUFWBQRiiJuKJMNup4fgylShKDWgRQ8sDNXIfNR4s68lkUZXbhMkKYo%2BCpWM8B4IomJiQXwggCrxQSTiTOiImfc59fc21iLEAF03NuZYgHvXwC2A8DcEWDh%2B4DBXCHncliWRMgVqztgYSGCYDADSRIgmwQA%2BT9gAqsupus8xtVnQVEE5Ub9uvjTgP2XH4TJTUAOqYWxG6FN%2Fu50WlfbCcEXH2tA53gE%2FZNBXLkzh7RmPH8vtwlor7Cjs7oY9aUZlMd%2FsLXA%2BrWVm371e9RYIFRVlSQSK5qxG6Fs7RQFu0XCbz9Zjk9ndFy9MYH%2B8TAGphbRO76w0aMRimQeneVW7PE58HCbD8VFNgDA9NiNrPYQfZ1JyxcBsRoIvw0snthcjhCEw1E1HI4NRSLLhzs6Gr5R8LdBhqHR1VqJrtZKxBNJXHtvGn3jIVwcmgXDUHio0o49Pjs%2B1lyGyorSwgwaenP1ZxNEo4lMMBiZjsXiJ6amhv7q4MGDqbU2lmV37pyXJRGPdNbgkc4afDIQQjSZxoN7mrad7RlBJpMh09PBYCwWP5NIhL7U3t6%2BYWRgg4sr30kmyRNmRfCaRG7H2PC6HYgntKyTp2ja0LP2RlCJGYtqcyyW8V2OEvvh9kbnsWwybHu7%2Fw8AoKenv0YUhH8SBP6AovDFgvDhfEBGSG7XwgQswlqDGlZ3DUUy1Yc79h38Ri7yd2fZ2dk8AuB3AKCvb%2BAphuH%2BTuDZVptNyvHe64NBNLMrs5ium4zTta8vxJQX9%2B%2Ffn9f3dBsuc0tL40kAJwGgt7f%2FRZZlX4xGkyazWdg0czQChsl%2FewNAQneToNoWj1HF40HuC4%2B1PvhrWb6ny46s%2B3zPnuZXALxy9OhRvra24WVJkj6jmIUqSeJzjheZTO6T1yAjmG6KxzJVlyNq%2BVc6H37K0P8BGEVeq7FRvJiZWYTHU7SlXDyhw%2Bt1bdlnenwAnuUX7vp1DJXfbHvw6a%2FnY6cRbPs86usb%2BA2a5v%2BaprS9LpeDpreoQbIREIlEyfzsWDwdGTnU2PXJv9%2FO%2FwIZRUEP5CtXBv9ElrkvmkS22Wo1cevjxUYErKwkSTgcC0Yi0QtA8o%2F8fn%2BgkDZlw45kJPfEC0Wokkyr8WKNAE3LIBiMxGOx1OVodOnl9nb%2F2Z2wwwh2PCW7detWvapSXxMFfh8BMQPcQDQaPtTe3vxfOz22EfwftIPf0Wg3pZwAAAAASUVORK5CYII%3D"></a>
<a href="https://hub.docker.com/r/irvlutd/rpx"><img alt="Docker images" src="https://img.shields.io/badge/docker-irvlutd%2Frpx-2496ED?logo=docker&logoColor=2496ED"></a>
<a href="https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests&logo=github"></a>
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-3da639"></a>

<br>

<img src="benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" width="95%" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires.">

<sub>The same real scenes before, during and after manipulation, scored on six perception tasks.</sub>

</div>

<br>

RPX is a benchmark and toolkit for robot perception in scenes that a robot is changing. We recorded
100 real scenes before (Clutter), during (Interaction) and after (Clean) manipulation, plus an
egocentric view, and score six tasks on the same frames. Two diagnostics summarize every model:
phase robustness Φ and worst-phase quality 𝒥<sub>min</sub>.

<div align="center">

| **0 of 65** | **2–4×** | **5–11×** | **0.23** |
|:-:|:-:|:-:|:-:|
| evaluations both accurate<br>and robust across phases | near-range video depth<br>error during Interaction | more tracker identity<br>switches during Interaction | best worst-phase score<br>for in-context grounding |

<sub>Results across 49 models and 65 evaluations in the paper.</sub>

<br>

[Citation](#citation) &nbsp;·&nbsp;
[Getting started](#getting-started) &nbsp;·&nbsp;
[Tasks](#tasks) &nbsp;·&nbsp;
[Repository layout](#repository-layout) &nbsp;·&nbsp;
[Documentation](#documentation)

</div>

## Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across
            Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and
            Kamath, Srinanditha and Allu, Sai Haneesh and Rangappa, Govind Tyagi and
            Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

## Getting started

The `rpx-benchmark` package provides the data, an adapter per task, the paper's benchmark
protocol, and the Φ and 𝒥<sub>min</sub> calculator. You write one function that runs your model.

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

The example below scores an image-depth model on a 6-frame trial. The download fetches only this
task's files (about 160 MB) and needs no login.

```python
import numpy as np
import rpx_benchmark as rpx

# 1. Download the task's data
manifest = rpx.download_split("monocular_depth", "easy", max_samples=6)

# 2. Your model: RGB image (H, W, 3) in, depth (H, W) in metres out
def predict(rgb):
    return np.ones(rgb.shape[:2], np.float32)

# 3. Wrap it in the task's adapter and run the benchmark
model = rpx.make_numpy_depth_model(predict, name="my-model")
result, report, paths = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(
    model=model, manifest_path=str(manifest), output_dir="rpx_results/my-model",
))
print(result.aggregated["absrel"], result.aggregated["delta1"])
```

Φ and 𝒥<sub>min</sub> compare phases, so they need all three phases of many scenes. Drop
`max_samples` to download the full tier, run the benchmark again, then:

```python
from rpx_benchmark.examples.summarize_metrics import calculate, read_records

metrics = ["absrel", "rmse", "silog", "delta1"]
summary = calculate(read_records(paths["json"]), metrics)
```

To try any task before writing code, `rpx.benchmark("T2", rpx.download("T2", max_samples=2))`
runs it with a placeholder model.

## Tasks

Every task follows the steps above with its own download name, adapter and runner.

| Task | Download | Adapter and runner | Paper metrics |
|:--|:--|:--|:--|
| T1 Image depth | `"monocular_depth"` | `make_numpy_depth_model`<br>`run_monocular_depth` | AbsRel, RMSE, SILog, δ1 |
| T2 Video depth | `"video_depth"` | `make_numpy_video_depth_model`<br>`run_video_depth` | T1's, plus TGM, TGSE |
| T3 Object tracking | `"object_tracking"` | `make_numpy_tracking_model`<br>`run_object_tracking` | HOTA, MOTA, IDSW |
| T4 Relative camera pose | `"relative_camera_pose"`* | `make_numpy_pose_model`<br>`run_relative_pose` | AUC at 5°, 10°, 20° |
| T5 VQA | `rpx.download("T5")` | `vqa.evaluate_vqa` | Acc@0.5, GIoU, center-in-GT |
| T6 In-context VQA | `rpx.download("T6")` | `vqa.evaluate_vqa` | Acc@0.5, GIoU, center-in-GT |

<sub>* Pass `manifest_name="relative_pose"` as well. Functions are under `rpx`; `vqa` is
`rpx_benchmark.vqa`, and its models return text, so they need no adapter.
[`dummy_models.py`](benchmark/rpx_benchmark/examples/dummy_models.py) shows the inputs and
outputs of every task.</sub>

To add a task or a metric, see the toolkit docs on
[tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) and
[metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/).

## Repository layout

| Folder | Contents |
|:--|:--|
| [`benchmark/`](benchmark/README.md) | the `rpx-benchmark` package, its tests and the docs source |
| [`docker/`](docker/README.md) | images with the models evaluated in the paper |
| [`data/capture/`](data/capture/README.md) | capture rig and recording scripts |
| [`data/mask_annotation/`](data/mask_annotation/README.md) | the instance-mask annotation pipeline |
| [`tracking/`](tracking/README.md) | text prompts for text-prompted trackers (T3) |
| [`tools/`](tools/dataset/README.md) | dataset card, dataset QA, docs and figure builders |

## Documentation

- [Toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/): model contracts per task, the evaluation protocol, Φ and 𝒥<sub>min</sub>, annotation tools
- [Dataset](https://huggingface.co/datasets/IRVLUTD/RPX): 100 scenes, 70 objects and 133K frames on Hugging Face
- [Docker images](docker/README.md): which image holds which of the paper's models
- [Python package](benchmark/README.md): the `rpx-benchmark` API, also on [PyPI](https://pypi.org/project/rpx-benchmark/)

---

<div align="center">
<sub>
Code under the <a href="LICENSE">MIT License</a>, dataset under CC BY 4.0.
Questions: <a href="https://github.com/IRVLUTD/RPX/issues">open an issue</a> or contact
<a href="https://jishnujayakumar.github.io/">Jishnu</a>,
<a href="https://github.com/Optimax14">Itay</a> or
<a href="https://narendhiranv04.github.io">Naren</a>.
<br>
Built at the <a href="https://labs.utdallas.edu/irvl/">Intelligent Robotics and Vision Lab</a>,
The University of Texas at Dallas.
</sub>
</div>
