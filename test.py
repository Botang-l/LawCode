import tool
#tool.LoadData(Jcase='建', OnlyJudge=True)
tool.AnalysisData('民法第495條,第四九五,四百九十五,民法第227-2條,第227條之2,227條第2,二二七之二,二二七條之二,情勢變更,情事變更,總價,單價')
tool.DeleteNull_and_OneHotEncoding()
tool.MachineLearning()