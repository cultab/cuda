library(tidyr)
library(ggplot2)
library(ggthemes)
library(tibble)
library(dplyr)
library(patchwork)
library(scales)
library(forcats)

theme_set(theme_clean())
# theme_set(theme_gdocs())

data1 <- read.csv("./results_1.csv", header = TRUE)
data2 <- read.csv("./results_2.csv", header = TRUE)
raw_data <- rbind(data1, data2)

# data <- gather(threads, ) %>%
# data <- gather(data, variable, value, sddm_time, diag_max_time, create_time, b_min_reduce_time) %>%
# data <- pivot_longer(data, c(sddm_time, diag_max_time, create_time, b_min_reduce_time, b_min_critical_time, b_min_tree_time), names_to = "variable", values_to = "value") %>%

# threads <- gather(data, variable, value, sddm_time, diag_max_time, create_time, b_min_reduce_time) %>%
# data <- gather(data, variable, value, limit) %>%
#     select(variable, value, threads)

# data <- data[data$elements > 10000, ]

facet_labs <- c(
    "radix" = "Radix sort",
    "bitonic" = "Bitonic sort",
    "counting" = "Counting sort",
    "1050ti" = "1050ti",
    "RTX TITAN" = "RTX TITAN",
    "16x16" = "16x16",
    "8x32" = "8x32",
    "16x64" = "16x64",
    "8x128" = "8x128",
    "96x128" = "96x128",
    "192x128" = "192x128",
    "1x256" = "1x256",
    "4x256" = "4x256",
    "1x1024" = "1x1024",
    "4x1024" = "4x1024",
    "192x1024" = "192x1024",
    "2" = "2",
    "16" = "16",
    "64" = "64",
    "256" = "256",
    "1024" = "1024"
)

# my_theme <- theme(
#     # legend.position = c(.05, .85),
#     legend.position = "right",
#     legend.key.size = unit(.5, "cm"),
#     legend.text = element_text(size = 7)
# )
# , labeller = as_labeller(facet_labs)

raw_data$method <- factor(raw_data$method)
raw_data$size <- factor(raw_data$size)
raw_data$blocks <- factor(raw_data$blocks)
raw_data$threads <- factor(raw_data$threads)
raw_data$gpu <- factor(raw_data$gpu)
raw_data$block_threads <- with(raw_data, interaction(blocks, threads, sep = "x"))

data1 <- raw_data[raw_data$max_value %in% c(256, -1), ]
data1$max_value <- factor(data1$max_value)


graph_all <- ggplot(data = data1,
                    aes(x = size, y = time, color = method, group = method)) +
    facet_grid(block_threads ~ gpu, labeller = as_labeller(facet_labs)) +
    # facet_grid(gpu ~ block_threads) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    theme(axis.text.x = element_text(
        angle = 45,
        vjust = 1,
        hjust = 1,
        size = 7)
    ) +
    labs(y = "Time log(s)", x = "Size of Array",
        color = "Sorting algorithm", title = "GPUs vs Block and Thread count")
    # my_theme +
    # hahaha theme(legend.position = c(0.84, 0.27))

data2 <- raw_data[raw_data$max_value %in% c(256, -1), ]
data2 <- data2[data2$block_threads %in% c("16x16", "192x1024"),]

graph_blocks <- ggplot(data = data2,
                    aes(x = size, y = time, color = method, group = method)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(gpu ~ block_threads, labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    theme(axis.text.x = element_text(
        angle = 45,
        vjust = 1,
        hjust = 1,
        size = 7)
    ) +
    labs(y = "Time log(s)", x = "Size of Array",
        color = "Sorting algorithm", "Block and Thread count vs GPUs")

data21 <- raw_data[raw_data$max_value %in% c(256, -1), ]
# data21 <- data21[data21$block_threads %in% c("16x16", "192x1024"),]

graph_blocks1 <- ggplot(data = data21,
                    aes(x = size, y = time, color = block_threads, group = block_threads)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(method ~ gpu, labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    theme(axis.text.x = element_text(
        angle = 45,
        vjust = 1,
        hjust = 1,
        size = 7)
    ) +
    labs(y = "Time log(s)", x = "Size of Array",
        color = "BLOCKSxTHREADS", title = "Algorithms vs GPUs")

data3 <- raw_data[raw_data$method == "radix", ]

graph_radix <- ggplot(data = data3,
                    aes(x = size, y = time, color = gpu, group = gpu)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(block_threads ~ ., labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    labs(y = "Time log(s)", x = "Size of Array",
        color = "GPU", title = "Radix sort")

data4 <- raw_data[raw_data$method == "counting", ]
data4$max_value <- factor(data4$max_value)

graph_counting <- ggplot(data = data4,
                    aes(x = size, y = time, color = gpu, group = gpu)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(block_threads ~ max_value, labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    theme(axis.text.x = element_text(
        angle = 45,
        vjust = 1,
        hjust = 1,
        size = 7)
    ) +
    labs(y = "Time log(s)", x = "Size of Array",
        color = "GPU", title = "Counting sort with different max values")

data5 <- raw_data[raw_data$method == "bitonic", ]

graph_bitonic <- ggplot(data = data5,
                    aes(x = size, y = time, color = gpu, group = gpu)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(block_threads ~ ., labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    labs(y = "Time log(s)", x = "Size of Array",
        color = "GPU", title = "Bitonic sort")

pdf <- TRUE

if (pdf)
    pdf("graphs.pdf", width = 10, height = 10)
print(graph_all)
print(graph_blocks)
print(graph_blocks1)
print(graph_radix)
print(graph_counting)
print(graph_bitonic)
print(graph_bitonic)
if (pdf)
    dev.off()
