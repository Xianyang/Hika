import xlsxwriter

a = 0.300
print '%f %%' % (a * 100)

workbook = xlsxwriter.Workbook('test.xlsx')
worksheet = workbook.add_worksheet()

percentAndBold = workbook.add_format({'num_format': '#.#0%', 'bold': True})
percent = workbook.add_format({'num_format': '#.#0%'})
bold = workbook.add_format({'bold': True})

formatForTitle = workbook.add_format({'diag_type': 2, 'text_wrap': True})
worksheet.set_column(0, 0, 22)
worksheet.set_row(0, 30)
# worksheet.write(0, 0, '                       position level\ntake profit', formatForTitle)
redAndYellow = workbook.add_format({'font_color': 'red', 'bg_color': 'yellow'})
worksheet.write(0, 0, 'a', redAndYellow)

workbook.close()