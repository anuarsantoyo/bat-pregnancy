# Content of the Drive Folder ML_bat_repstat

The folder contains raw data and processed logger data of bat activity a RFID-monitored day roosts from three colonies (BS, GB2, UA) of *Myotis bechsteinii* located close to the city of Würzburg, Germany from 2008 to 2024.

The folder contains the following directories:

```
- ML_bat_repstat

    - Data_2017 - Raw logger data without any processing from each colony in csv format
        - BS_2017
            - BS_Mbec_2017 - Raw data for *Myotis bechsteinii* from BS
            - BS_Mnat_2017 - Raw data for *Myotis nattereri* from BS
            - BS_Paur_2017 - Raw data for *Plecotus auritus* from BS
        - GB2_2017
            - GB2_Mbec_2017 - Raw data for M. bechstenii from GB2
            - GB2_Paur_2017 - Raw data for *Plecotus auritus* from GB2
        - UA_2017
            - UA_Mbec_2017  - Raw data from M. bechsteinii from UA

    - Data_2018 - Raw logger data without any processing from each colony
        - BS_2018
            - BS_Mbec_2018 - Raw data for *Myotis bechsteinii* from BS
            - BS_Mnat_2018 - Raw data for *Myotis nattereri* from BS
            - BS_Paur_2018 - Raw data for *Plecotus auritus* from BS
        - GB2_2018
            - GB2_Mbec_2018 - Raw data for M. bechstenii from GB2
            - GB2_Paur_2018 - Raw data for *Plecotus auritus* from GB2
        - UA_2018
            - UA_Mbec_2018  - Raw data from M. bechsteinii from UA
            
    - yearly - Yearly M. bechsteinii activity processed data derived from raw logger data from 2008 to 2024 organized by year --> colony. The following subdirectory is an example from 2010 that replicate its content in each year subdirectory. The processed files has the following name structure YYYY_Colony_lid_sitDDs.csv the last digits indicate how many seconds between two consecutive readings from the same individual were taken into account to consider an independent visit event.
        - 2008
        - 2009
        - 2010
            - BS
                - 2010_BS_lid_sit30s.csv - time span between readings 10 seconds
                - 2010_BS_lid_sit30s.csv
                - 2010_BS_lid_sit60s.csv
                - 2010_BS_lid_sit120s.csv
                - 2010_BS_lid_sit300s.csv
                - 2010_BS_lid_sit600s.csv
            - GB2
            - UA
        .
        .
        .
        - 2024
        
    - yearly_table_L_Mbec.xlsx - file with reproductive data, see sheet "Dictionary" for columns meaning. The column 'birth_event' encodes whether the bat was pregnant based on genetic analysis. 
    
```