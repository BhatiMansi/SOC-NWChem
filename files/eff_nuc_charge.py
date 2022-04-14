def Zeff(Zeff):
    if Zeff == 1:
        Z = {'H': 1.00, 
             'He': 2.00,
             'Li': 1.35,
             'Be': 2.00,
             'B': 2.75,
             'C': 3.60,
             'N': 4.55,
             'O': 5.60,
             'F': 6.75,
             'Ne': 8.00,
             'Na': 10.04,
             'Mg': 10.80,
             'Al': 11.53,
             'Si': 12.25,
             'P': 12.94,
             'S': 13.60,
             'Cl': 14.24,
             'Ar': 14.85
             }

    elif Zeff == 2:

        Z = {'H': 1.00, 
             'He': 2.00,
             'Zn': 330,
             'La': 803.7,
             'Cd': 1584,
             'Hg': 9040
            }

        for i,ele in enumerate(['Li','Be','B','C','N','O','F','Ne']):
            Z[ele] = (0.2517 + 0.0626*(i+1))*(i+3)

        for i,ele in enumerate(['Na','Mg','Al','Si','P','S','Cl','Ar']):
            Z[ele] = (0.7213 + 0.0144*(i+1))*(i+11)

        for i,ele in enumerate(['K','Ca','Ga','Ge','As','Se','Br','Kr']):
            if i<2:
                Z[ele] = (0.8791 + 0.0039*(i+1))*(i+19)
            else:
                Z[ele] = (0.8791 + 0.0039*(i+1))*(i+29)

        for i,ele in enumerate(['Rb','Sr','In','Sn','Sb','Te','I','Xe']):
            if i<2:
                Z[ele] = (0.9228 + 0.0017*(i+1))*(i+37)
            else:
                Z[ele] = (0.9228 + 0.0017*(i+1))*(i+47)

        for i,ele in enumerate(['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu']):
            Z[ele] = (0.385 + 0.025*(i+1))*(i+21)

        for i,ele in enumerate(['Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag']):
            Z[ele] = (4.68 + 0.06*(i+1))*(i+39)

        for i,ele in enumerate(['Hf','Ta','W','Re','Os','Ir','Pt','Au']):
            Z[ele] = (13.96 + 0.14*(i+2))*(i+72)

    return Z
