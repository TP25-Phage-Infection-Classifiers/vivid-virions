library("DESeq2")
library(ggplot2)

# Load dataset
counts <- read.csv("data/guegler_2021/Guegler_T4_minusToxIN_full_raw_counts.tsv", sep="\t")
colnames(counts) <- gsub("^X", "", colnames(counts))

# Split into host and phage sets
host_counts <- counts[which(counts$Entity == "host"), 2:13]  # Columns 2–13 are sample counts
rownames(host_counts) <- counts$Geneid[which(counts$Entity == "host")]
head(host_counts)

phage_counts <- counts[which(counts$Entity == "phage"), 2:13]
rownames(phage_counts) <- counts$Geneid[which(counts$Entity == "phage")]
phage_counts <- phage_counts[, -c(1,2)]
# phage_counts <- counts[, 2:13]
# rownames(phage_counts) <- counts$Geneid
head(phage_counts)

# Create metadata
samples <- colnames(phage_counts)
timepoints <- gsub("_R[12]", "", samples)  # Remove R1/2 from timepoints
run <- gsub(".*_(R[12])", "\\1", samples)  # Add run number

colData <- data.frame(
  row.names = samples,
  time = factor(timepoints, levels = c("0", "2.5", "5", "10", "20", "30")),  # DSEq requires a factor
  run = run
)
#View(colData)

dds <- DESeqDataSetFromMatrix(countData = phage_counts,
                              colData = colData,
                              design = ~ time)

dds <- DESeq(dds)
norm_counts <- counts(dds, normalized = TRUE)
View(norm_counts)
#write.csv(norm_counts, "normalized_counts_phage_T4.csv")


# Boxplot for raw counts
boxplot(log2(phage_counts + 1), 
        las = 2, 
        main = "Raw Counts (log2)", 
        col = "lightgray", 
        ylab = "log2(count + 1)")

# Boxplot for normalized counts
boxplot(log2(norm_counts + 1), 
        las = 2, 
        main = "Normalized Counts (log2)", 
        col = "lightblue", 
        ylab = "log2(normalized count + 1)")


vsd_phage <- varianceStabilizingTransformation(dds, fitType="local") 
plotPCA(vsd_phage, intgroup = "time")
