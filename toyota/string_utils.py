def apply_string_treatments(df, cols):
    """
        Apply different string operations to clean/standarize the columns.
    """
    
    if type(cols) != list:
        raise Exception("'cols' must be a list.");
        
    if "Model" in cols:
        # ------ CHARACTERS ------
        df['Model'] = df['Model'].str.upper()
        df['Model'] = df['Model'].str.strip()
        df['Model'] = df['Model'].str.replace('?', '')
        df['Model'] = df['Model'].str.replace('!', ' ')
        df['Model'] = df['Model'].str.replace('+', ' ')
        df['Model'] = df['Model'].str.replace('MODEL', ' ')
        df['Model'] = df['Model'].str.replace('TOYOTA COROLLA', ' ')
        df['Model'] = df['Model'].str.replace('LINEA', ' ')
        df['Model'] = df['Model'].str.replace('LIN.', ' ')
        df['Model'] = df['Model'].str.replace('L.', ' ')
        df['Model'] = df['Model'].str.replace('BLUE', ' ') # Color azul
        df['Model'] = df['Model'].str.replace('(NW TYPE)', ' ') # Modelo nuevo
        df['Model'] = df['Model'].str.replace(' NW ', ' ') # Modelo nuevo
        df['Model'] = df['Model'].str.replace('NIEUW', ' ') # Modelo nuevo
        df['Model'] = df['Model'].str.replace('(7)', ' ')
        df['Model'] = df['Model'].str.replace(' 1 ', ' ') 
        df['Model'] = df['Model'].str.replace('13I', ' ') 
        df['Model'] = df['Model'].str.replace('1 6-16v', ' 16V ') 

        # Doors
        df['Model'] = df['Model'].str.replace(r' \d+\/\d+\-DOORS', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+\-DOORS', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+\-DRS', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+DRS', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+ DRS', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+DR', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+D ', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+ DOORS', ' ', regex=True)
        df['Model'] = df['Model'].str.replace(r' \d+DOORS', ' ', regex=True)


        # ----- MODEL NAME WORDS ------
        # Hatchback
        df['Model'] = df['Model'].str.replace('TERRA/COMF.', ' TERRA COMFORT ')
        df['Model'] = df['Model'].str.replace('G6-R', ' G6 ')
        df['Model'] = df['Model'].str.replace('G6R', ' G6 ')
        df['Model'] = df['Model'].str.replace('*G3*', ' G3 ')
        df['Model'] = df['Model'].str.replace('HATCHBACK', ' HATCH_B ')
        df['Model'] = df['Model'].str.replace('HATCHB', ' HATCH_B ')
        df['Model'] = df['Model'].str.replace('HB', ' HATCH_B ')
        df['Model'] = df['Model'].str.replace(' H ', ' HATCH_B ')

        # Liftback
        df['Model'] = df['Model'].str.replace('LIFTBACK', ' LIFTB ')
        df['Model'] = df['Model'].str.replace('LIFT ', ' LIFTB ')
        df['Model'] = df['Model'].str.replace('LB', ' LIFTB ')

        # Es 'Sport'
        df['Model'] = df['Model'].str.replace('T SPORT', ' SPORT ')
        df['Model'] = df['Model'].str.replace('T-SPORT', ' SPORT ')
        df['Model'] = df['Model'].str.replace(' S ', ' SPORT ')

        # Wagon
        df['Model'] = df['Model'].str.replace('WGN', ' WAGON ')
        df['Model'] = df['Model'].str.replace('STATIONWAGEN', ' WAGON ')
        df['Model'] = df['Model'].str.replace('STION', ' WAGON ')
        df['Model'] = df['Model'].str.replace('STATION', ' WAGON ')
        df['Model'] = df['Model'].str.replace('ST ', ' ')

        # Sedan
        df['Model'] = df['Model'].str.replace('SDN', ' SEDAN ')
        df['Model'] = df['Model'].str.replace('SD', ' SEDAN ')
        df['Model'] = df['Model'].str.replace('COMF.', ' COMFORT ')
        df['Model'] = df['Model'].str.replace('COMF ', ' COMFORT ')

        # Other models
        df['Model'] = df['Model'].str.replace('ANDERS', ' COMM ')
        df['Model'] = df['Model'].str.replace('EXECUTIVE', ' EXEC ')

        # ----- SPECIFICATION WORDS ------
        # HP
        df['Model'] = df['Model'].str.replace('90', ' ')
        df['Model'] = df['Model'].str.replace('110', ' ')
        df['Model'] = df['Model'].str.replace('130', ' ')
        df['Model'] = df['Model'].str.replace('116', ' ')
        df['Model'] = df['Model'].str.replace('700', ' ')
        df['Model'] = df['Model'].str.replace('1800', ' ')

        # Combustible
        df['Model'] = df['Model'].str.replace(' D ', ' DSL ')
        df['Model'] = df['Model'].str.replace('DIESEL', ' DSL ')

        # Valvulas
        df['Model'] = df['Model'].str.replace('VVT I', ' VVTI ')
        df['Model'] = df['Model'].str.replace(' I ', ' VVTI ')
        df['Model'] = df['Model'].str.replace('VVT-I', ' VVTI ')
        df['Model'] = df['Model'].str.replace('VVTL-I', ' VVTLI ')
        df['Model'] = df['Model'].str.replace('VVTL I', ' VVTLI ')
        df['Model'] = df['Model'].str.replace('I-16V', ' 16V ')
        df['Model'] = df['Model'].str.replace('-16 V ', ' 16V ')
        df['Model'] = df['Model'].str.replace(' 16 V ', ' 16V ')
        df['Model'] = df['Model'].str.replace('-16', ' 16V ')
        df['Model'] = df['Model'].str.replace(' 16 ', ' 16V ')

        # CC
        df['Model'] = df['Model'].str.replace(r'\d+.\d+I', ' VVTI ', regex=True) # Iyeccion Inteligente
        df['Model'] = df['Model'].str.replace(r'\d+\.\d+D', ' DSL ', regex=True) # Diesel
        df['Model'] = df['Model'].str.replace(r'\d+\.\d+V', ' ', regex=True) # TODO Error de tipeo
        df['Model'] = df['Model'].str.replace(r'\d+\.\d', ' ', regex=True)

        # Transmision
        df['Model'] = df['Model'].str.replace('SPORTS TC', ' SPORT ') # Sports Transmission Control
        df['Model'] = df['Model'].str.replace('D4D', ' D4D ') 
        df['Model'] = df['Model'].str.replace('AUT4', ' MATIC4 ')
        df['Model'] = df['Model'].str.replace('AUT3', ' MATIC3 ')
        df['Model'] = df['Model'].str.replace('AUT.', ' MATIC ')
        df['Model'] = df['Model'].str.replace('AUT', ' MATIC ')

        # Airconditioner
        df['Model'] = df['Model'].str.replace('AIRCO', 'AIRCO')

        # Especials
        df['Model'] = df['Model'].str.replace('NAVIGATIE', ' NAV ')
        df['Model'] = df['Model'].str.replace('B.EDITION', ' B_ED ')
        df['Model'] = df['Model'].str.replace(' / ', ' ')

        
    # Dummy variables for Fuel_Type
    if "Fuel_Type" in cols:
        df["Petrol"] = df["Fuel_Type"].apply(lambda x: 1 if x == "Petrol" else 0)
        df["Diesel"] = df["Fuel_Type"].apply(lambda x: 1 if x == "Diesel" else 0)
        df["CNG"] = df["Fuel_Type"].apply(lambda x: 1 if x == "CNG" else 0)
        df.drop("Fuel_Type", axis=1, inplace=True)
    
    return df

def infer_new_model_columns(df):
    """
        Infer new columns from the "Model" column based on the 'relevant terms', to add more information to the dataset.
        A 'relevant term' is a single-word uppercase string. E.g: TERRA, LUNA, VVTLI, etc. 
        Note the 'm_' to indicate that the column is from "model".
    """

    word_counts = df['Model'].str.split().explode().value_counts().reset_index()
    word_counts.columns = ['Term', 'Value_Count']
    word_counts_list = list(zip(word_counts['Term'], word_counts['Value_Count']))

    print("Relevant terms and counts: ")
    for word in word_counts_list: 
        print(word)
        term = word[0]
        df[f"m_{term.lower()}"] = df['Model'].apply(lambda x: 1 if term in x else 0)  

    # # Simplificamos el TIPO DE TRANSMISION
    # def get_m_trans(row):
    #     if row['m_matic'] == 1:
    #         return 1
    #     elif row['m_matic3'] == 1:
    #         return 2
    #     elif row['m_matic4'] == 1:
    #         return 3
    #     else:
    #         return 0

    cols_to_drop = [
        'm_6', 'm_v', 'm_r', 
        'm_bns', 'm_exec', 'm_gtsi',
        'm_verso', 'm_mpv', 'm_comm',
        'm_gl', 'm_ll', 'm_nav', 'm_pk', 'm_so', 'm_xl',
        'm_sw', 'm_b_ed', 'm_g3',
        'm_keuze', 'm_uit', 'm_occasions', 'm_s-uitvoering']

    df.drop([col for col in cols_to_drop if col in df.columns], axis=1, inplace=True)

    return df


def get_new_cols(df):
    """
        It return the list of 'model' columns.
    """
    return [col for col in df.columns if col.startswith("m_")]
