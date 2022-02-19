library(tidyr)
library(ggplot2)
library(ggthemes)
library(tibble)
library(dplyr)
library(patchwork)
library(scales)
library(forcats)

theme_set(theme_clean())
# theme_set(theme_pander())

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
    "TITAN RTX" = "TITAN RTX",
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

my_theme <- theme(legend.title = element_text(size=9))
x_theme <- theme(
    axis.text.x = element_text(
        angle = 45,
        vjust = 1,
        hjust = 1,
        size = 7
    )
    # ,axis.title.y = element_text(
    #     size = 10
    #     vjust = 1.5
    # )
)

x_scale <- scale_x_discrete(breaks = c("10", "100", "1000", "10000", "100000", "1000000", "10000000", "100000000"))
y_scale <- scale_y_continuous(trans = "log2", labels = scientific)

raw_data$method <- factor(raw_data$method)
raw_data$size <- factor(raw_data$size)
raw_data$blocks <- factor(raw_data$blocks)
raw_data$threads <- factor(raw_data$threads)
raw_data$gpu <- factor(raw_data$gpu)
raw_data$block_threads <- with(raw_data, interaction(blocks, threads, sep = "x"))

data1 <- raw_data[raw_data$max_value %in% c(256, -1), ]
data1$max_value <- factor(data1$max_value)

data1 <- data1[data1$block_threads %in% c("16x16", "8x32", "8x128", "96x128", "192x1024"), ]


graph_all <- ggplot(data = data1,
                    aes(x = size, y = time, color = method, group = method)) +
    facet_grid(block_threads ~ gpu, labeller = as_labeller(facet_labs)) +
    # facet_grid(gpu ~ block_threads) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    y_scale +
    x_scale +
    labs(y = "Time (s)", x = "Size of Array",
        color = "Sorting algorithm", title = "GPUs vs Block and Thread count") +
    my_theme + x_theme

data2 <- raw_data[raw_data$max_value %in% c(256, -1), ]
data2 <- data2[data2$block_threads %in% c("16x16", "192x1024"),]

graph_blocks <- ggplot(data = data2,
                    aes(x = size, y = time, color = method, group = method)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(gpu ~ block_threads, labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    y_scale +
    x_scale +
    labs(y = "Time (s)", x = "Size of Array",
        color = "Sorting algorithm", title = "Block and Thread count vs GPUs") +
    my_theme + x_theme

data21 <- raw_data[raw_data$max_value %in% c(256, -1), ]
# data21 <- data21[data21$block_threads %in% c("16x16", "192x1024"),]

graph_blocks1 <- ggplot(data = data21,
                    aes(x = size, y = time, color = block_threads, group = block_threads)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(method ~ gpu, labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line(size=0.2) +
    geom_point(size=0.6) +
    y_scale +
    x_scale +
    labs(y = "Time (s)", x = "Size of Array",
        color = "Blocks x Threads", title = "Algorithms vs GPUs") +
    my_theme + x_theme

data3 <- raw_data[raw_data$method == "radix", ]
data3 <- data3[data3$block_threads %in% c("16x16", "8x32", "8x128", "96x128", "192x1024"), ]

graph_radix <- ggplot(data = data3,
                    aes(x = size, y = time, color = gpu, group = gpu)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(block_threads ~ ., labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    y_scale +
    labs(y = "Time (s)", x = "Size of Array",
        color = "GPU", title = "Radix sort") + my_theme + x_theme

data4 <- raw_data[raw_data$method == "counting", ]
data4$max_value <- factor(data4$max_value)
data4 <- data4[data4$block_threads %in% c("16x16", "8x32", "8x128", "96x128", "192x1024"), ]

graph_counting <- ggplot(data = data4,
                    aes(x = size, y = time, color = gpu, group = gpu)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(block_threads ~ max_value, labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line(size=0.2) +
    geom_point(size=0.6) +
    y_scale +
    x_scale +
    labs(y = "Time (s)", x = "Size of Array",
        color = "GPU", title = "Counting sort with different max values") +
    theme(
        axis.text.x = element_text(
            angle = 90,
            vjust = 0.5,
            hjust = 1,
            size = 7
        )
    ) + my_theme

data5 <- raw_data[raw_data$method == "bitonic", ]
data5 <- data5[data5$block_threads %in% c("16x16", "8x32", "8x128", "96x128", "192x1024"), ]

graph_bitonic <- ggplot(data = data5,
                    aes(x = size, y = time, color = gpu, group = gpu)) +
    # facet_grid(block_threads ~ gpu) +
    facet_grid(block_threads ~ ., labeller = as_labeller(facet_labs)) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    y_scale +
    labs(y = "Time (s)", x = "Size of Array",
        color = "GPU", title = "Bitonic sort") + my_theme + x_theme

pdf <- FALSE

if (pdf) {
    # pdf("graphs.pdf", width = 10, height = 10)
    pdf("graphs.pdf")
    print(graph_all)
    print(graph_blocks)
    print(graph_blocks1)
    print(graph_radix)
    print(graph_counting)
    print(graph_bitonic)
    dev.off()
}
