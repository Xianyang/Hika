import xlsxwriter

workbook = xlsxwriter.Workbook('test.xlsx')
worksheet = workbook.add_worksheet()

percentAndBold = workbook.add_format({'num_format': '#.#0%', 'bold': True})
percent = workbook.add_format({'num_format': '#.#0%'})
bold = workbook.add_format({'bold': True})

formatForTitle = workbook.add_format({'diag_type': 2, 'text_wrap': True})
worksheet.set_column(0, 0, 22)
worksheet.set_row(0, 30)
worksheet.write(0, 0, '                       position level\ntake profit', formatForTitle)

workbook.close()