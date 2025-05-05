library("DESeq2")
library("ggplot2")

# Import data
counts <- read.csv("data/wolfram-schauerte_2022/Wolfram-Schauerte_full_raw_counts.tsv", sep="\t")
# Set row names to gene ids
rownames(counts) <- counts$Geneid


# Remove unused columns
counts_rest <- counts[, c("Entity", "Symbol")]
counts <- counts[, -c(7, 8)]
colnames(counts) <- gsub("^X", "", colnames(counts))  # Entferne X von spaltennamen (keine Ahnung woher die kommen)

counts_phage <- counts[counts_rest$Entity == "phage", ]
counts_host <- counts[counts_rest$Entity == "host", ]

head(counts)


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
norm_phage <- counts(dds_phage, normalized = TRUE)
View(norm_phage)
norm_host <- counts(dds_host, normalized = TRUE)
write.csv(norm_host, "normalized_counts_host.csv")
write.csv(norm_phage, "normalized_counts_phage.csv")
write.csv(counts(dds_phage, normalized = FALSE), "non_normalized_counts_phage.csv")


## Visualize

# PCA plots showing variance
vsd_host <- vst(dds_host)
plotPCA(vsd_host, intgroup = "time")

vsd_phage <- varianceStabilizingTransformation(dds_phage)  # same as above but dds_phage has too few nrows
plotPCA(vsd_phage, intgroup = "time")



raw_long <- as.data.frame(counts(dds_phage, normalized = FALSE))
raw_long$gene <- rownames(raw_long)
raw_long <- tidyr::pivot_longer(raw_long, -gene, names_to = "sample", values_to = "count")

ggplot(raw_long, aes(x = sample, y = count)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "Raw counts per sample", y = "Log10 counts") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

norm_long <- as.data.frame(counts(dds_phage, normalized = TRUE))
norm_long$gene <- rownames(norm_long)
norm_long <- tidyr::pivot_longer(norm_long, -gene, names_to = "sample", values_to = "count")

ggplot(norm_long, aes(x = sample, y = count)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "Normalized counts per sample", y = "Log10 counts") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
