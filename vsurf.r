#install.packages("languageserver")
#install.packages("VSURF")

training_csv_path = "/mnt/SN750/00_Donnees_SSD/csv_rf/estrie_training_full.csv"

# RowsInCSV = 116391936
# List <- lapply(1:100, function(x) read.csv(training_csv_path, nrows=1, skip = sample(RowsInCSV, 1), header=FALSE))
# DF = do.call(rbind, List)

# library(sqldf)
# DF <- read.csv.sql(training_csv_path, sql = "select * from file order by random() limit 20000")


library(data.table)
#DF <- fread(training_csv_path, nrows = 20000)
col_names_names <- c('no_row', 's2_e_B1', 's2_e_B2', 's2_e_B3', 's2_e_B4', 's2_e_B5',
              's2_e_B6', 's2_e_B7', 's2_e_B8', 's2_e_B8a', 's2_e_B9', 's2_e_B11',
              's2_e_B12', 's2_p_B1', 's2_p_B2', 's2_p_B3', 's2_p_B4', 's2_p_B5',
              's2_p_B6', 's2_p_B7', 's2_p_B8', 's2_p_B8a', 's2_p_B9', 's2_p_B11',
              's2_p_B12', 's1_e_VH', 's1_e_VV', 's1_e_ratio', 's1_p_VH', 's1_p_VV',
              's1_p_ratio', 'mnt', 'mhc', 'slopes', 'tpi', 'tri', 'twi', 'mask_bin', 'mask_multi')

DF <- fread("shuf -n 100 /mnt/SN750/00_Donnees_SSD/csv_rf/estrie_training_full.csv", col.names=col_names_names)

x = DF[,  c('s2_e_B1', 's2_e_B2', 's2_e_B3', 's2_e_B4', 's2_e_B5',
        's2_e_B6', 's2_e_B7', 's2_e_B8', 's2_e_B8a', 's2_e_B9', 's2_e_B11',
        's2_e_B12', 's2_p_B1', 's2_p_B2', 's2_p_B3', 's2_p_B4', 's2_p_B5',
        's2_p_B6', 's2_p_B7', 's2_p_B8', 's2_p_B8a', 's2_p_B9', 's2_p_B11',
        's2_p_B12', 's1_e_VH', 's1_e_VV', 's1_e_ratio', 's1_p_VH', 's1_p_VV',
        's1_p_ratio', 'mnt', 'mhc', 'slopes', 'tpi', 'tri', 'twi')]

y = DF$mask_multi

library('VSURF')
df.vsurf <- VSURF(x=x, y=y, parallel=TRUE)