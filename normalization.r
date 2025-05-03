library("DESeq2")

# Import data
counts <- read.csv("data/wolfram-schauerte_2022/Wolfram-Schauerte_full_raw_counts.tsv", sep="\t")
# Set row names to gene ids
rownames(counts) <- counts$Geneid


# Remove unused columns
counts_rest <- counts[, c("Entity", "Symbol")]
counts <- counts[, -c(1, 16, 17)]
colnames(counts) <- gsub("^X", "", colnames(counts))  # Entferne X von spaltennamen (keine Ahnung woher die kommen)

counts_phage <- counts[counts_rest$Entity == "phage", ]
counts_host <- counts[counts_rest$Entity == "host", ]

head(counts_host)


# Metadata, für DESeqDataSetFromMatrix gebraucht
coldata <- read.csv("data/wolfram-schauerte_2022/Wolfram-Schauerte_SraRunTable.csv")
id_col <- c("20_R3", "20_R2", "20_R1",
            "7_R3", "7_R2", "7_R1", 
            "4_R3", "4_R2", "4_R1", 
            "1_R3", "1_R2", "1_R1",
            "0_R3", "0_R2", "0_R1")  
coldata <- cbind(id = id_col, coldata)  # Weise gleiche ids wie in counts zu
head(coldata)
rownames(coldata) <- coldata$id
coldata <- coldata[, c("Run", "time"), drop = FALSE]  # Lösche alle spalten außer Run und time
coldata <- coldata[!(rownames(coldata) == "4_R1"), ]  # Diese spalte fehlt im datensatz, aber nicht in SraRunTable
coldata <- coldata[nrow(coldata):1, ]
coldata$time <- factor(coldata$time)  # Next step brauch factor anstatt normalem wert
coldata$Run <- factor(coldata$Run)

head(coldata)


# Normalization Host/Phage seperat

dds_phage <- DESeqDataSetFromMatrix(countData = counts_phage, colData = coldata, design = ~time)
dds_phage <- DESeq(dds_phage)

dds_host <- DESeqDataSetFromMatrix(countData = counts_host, colData = coldata, design = ~time)
dds_host <- DESeq(dds_host)

# Get normalized counts
norm_host <- counts(dds_host, normalized = TRUE)
norm_phage <- counts(dds_phage, normalized = TRUE)
write.csv(norm_host, "normalized_counts_host.csv")
write.csv(norm_phage, "normalized_counts_phage.csv")


vsd <- vst(dds)
plotPCA(vsd, intgroup = "time")

res <- results(dds)
res
