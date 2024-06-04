# Load libraries
library(dplyr)
library(ggplot2)
library(reshape2)
library(readr)


# Load data
setwd("C:/Users/Katie Overengland/Desktop/STATS 527")
data <- read_csv("sep23small.csv")

# Convert seconds columns to times of day (in hours)
data$start_time <- data$start_seconds / (60 * 60)
data$end_time <- data$end_seconds / (60 * 60)

# Normalize latitudes and longitudes
data$start_lat_r <- (data$start_lat - min(data$start_lat, data$end_lat)) / (max(data$start_lat, data$end_lat) - min(data$start_lat, data$end_lat))
data$start_lng_r <- (data$start_lng - min(data$start_lng, data$end_lng)) / (max(data$start_lng, data$end_lng) - min(data$start_lng, data$end_lng))
data$end_lat_r <- (data$end_lat - min(data$start_lat, data$end_lat)) / (max(data$start_lat, data$end_lat) - min(data$start_lat, data$end_lat))
data$end_lng_r <- (data$end_lng - min(data$start_lng, data$end_lng)) / (max(data$start_lng, data$end_lng) - min(data$start_lng, data$end_lng))


# Explore data
ggplot(data) +
  geom_point(aes(y = start_lat_r, x = start_lng_r))

ggplot(data) +
  geom_point(aes(y = end_lat_r, x = end_lng_r))

ggplot(data) +
  geom_point(aes(y = end_time, x = start_time))


############## TIME ################

# Define dense time grid onto which we evaluate our KDE
t_grid <- seq(min(data$start_time), 
              max(data$start_time), length = 96)

# Define the 1D Gaussian Kernel Function
gaussian_kernel_1d <- function(x) {
  exp(- x^2 / 2) / sqrt(2 * pi)
}

# Define custom temporal distance (wrap-around)
d_time <- function(t1, t2) {
  min(abs(t1 - t2), 24 - abs(t1 - t2))
}

## Compute KDE for t (start time)

# Bandwidth (Smoothing parameter)
h_3 = 1

# Initialize vector to store density estimates of t
density_t <- matrix(0, nrow = length(t_grid), ncol = 1)

# Calculate density estimate for each grid point
N <- nrow(data)
for (i in 1:length(t_grid)) {
  
  # Sum the Gaussian kernel contributions from each data point
  for (k in 1:N) {
    density_t[i] <- density_t[i] + gaussian_kernel_1d((t_grid[i] - data$start_time[k]) / h_3) / (h_3 * N)
  }
}

# Plot density estimate of t (start time)
ggplot(NULL) +
  geom_line(aes(y = density_t, x = t_grid)) +
  labs(x="start_time",
       y="estimated density",
       title = "No wrap-around") +
  theme(text = element_text(family = 'Fira Sans'),
        plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank())

# Now modify to incorporate wrap-around

# Initialize vector to store density estimates of t
density_t2 <- matrix(0, nrow = length(t_grid), ncol = 1)

# Calculate density estimate for each grid point
N <- nrow(data)
for (i in 1:length(t_grid)) {
  
  # Sum the Gaussian kernel contributions from each data point
  for (k in 1:N) {
    density_t2[i] <- density_t2[i] + gaussian_kernel_1d(d_time(t_grid[i], data$start_time[k]) / h_3) / (h_3 * N)
  }
}

# Plot density estimate of t (start time)
ggplot(NULL) +
  geom_line(aes(y = density_t2, x = t_grid)) +
  labs(x="start_time",
       y="estimated density",
       title = "1D KDE with Gaussian, h=1") +
  scale_x_continuous(breaks = seq(0, 24, by = 4)) +
  theme(text = element_text(family = 'Fira Sans'),
        plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank())


############## SPACE ################

# First, let's replicate a simpler approach:
# use a 2D Gaussian kernel to estimate the 
# distribution of start lats and lons,
# using a simple notion of distance

# Define dense grid onto which we evaluate our KDEs
x_grid <- seq(min(data$start_lng_r), max(data$start_lng_r), length = 30)
y_grid <- seq(min(data$start_lat_r), max(data$start_lat_r), length = 70)

# Define the 2D Gaussian Kernel Function
gaussian_kernel_2d <- function(x, y) {
  gaussian_kernel_1d(x) * gaussian_kernel_1d(y)
}

### Compute KDE for the start_lat, start_lng

# Bandwidths (Smoothing parameter)
h_1 = 0.08
h_2 = 0.05

# Initialize matrix to store density estimates of (X,Y)
density_xy <- matrix(0, nrow = length(x_grid), ncol = length(y_grid))

# Calculate density estimate of (Y,X) for each grid point
N <- nrow(data)
for (i in 1:length(x_grid)) {
  for (j in 1:length(y_grid)) {
    # Sum the Gaussian kernel contributions from each data point
    for (k in 1:N) {
      u <- (x_grid[i] - data$start_lng_r[k]) / h_1
      v <- (y_grid[j] - data$start_lat_r[k]) / h_2
      density_xy[i, j] <- density_xy[i, j] + gaussian_kernel_2d(u, v)
    }
    # Normalize by bandwidth and number of data points
    density_xy[i, j] <- density_xy[i, j] / (h_1 * h_2 * N)
  }
}

# Plot density estimate of start_lat, start_lng
# Unfortunately ggplot2 doesn't allow us to plot a matrix directly,
# so we need to convert in the form of (value, position_x, position_y)
density_xy_coord <- melt(density_xy, value.name = "value")
density_xy_coord$lng_grid = x_grid[density_xy_coord$Var1]
density_xy_coord$lat_grid = y_grid[density_xy_coord$Var2]

# Plot: 2D Gaussian KDE
ggplot(density_xy_coord, aes(density_xy_coord$lng_grid, density_xy_coord$lat_grid)) +
  geom_raster(aes(fill = density_xy_coord$value), show.legend =  FALSE) +
  scale_fill_distiller(palette = "Spectral", direction = -1) +
  labs(x="start_lng",
       y="start_lat",
       title = bquote("2D KDE with Gaussian"),
       fill = "density") +
  theme(text = element_text(family = 'Fira Sans'),
        plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank(),
        panel.background = element_rect(fill = "transparent"))


# Now do the same thing except for end locations

# Define dense grid onto which we evaluate our KDEs
x_grid <- seq(min(data$end_lng_r), max(data$end_lng_r), length = 30)
y_grid <- seq(min(data$end_lat_r), max(data$end_lat_r), length = 70)

# Initialize matrix to store density estimates of (X,Y)
density_xy2 <- matrix(0, nrow = length(x_grid), ncol = length(y_grid))

# Calculate density estimate of (Y,X) for each grid point
N <- nrow(data)
for (i in 1:length(x_grid)) {
  for (j in 1:length(y_grid)) {
    # Sum the Gaussian kernel contributions from each data point
    for (k in 1:N) {
      u <- (x_grid[i] - data$end_lng_r[k]) / h_1
      v <- (y_grid[j] - data$end_lat_r[k]) / h_2
      density_xy2[i, j] <- density_xy2[i, j] + gaussian_kernel_2d(u, v)
    }
    # Normalize by bandwidth and number of data points
    density_xy2[i, j] <- density_xy2[i, j] / (h_1 * h_2 * N)
  }
}

# Plot density estimate of start_lat, start_lng
# Unfortunately ggplot2 doesn't allow us to plot a matrix directly,
# so we need to convert in the form of (value, position_x, position_y)
density_xy_coord2 <- melt(density_xy2, value.name = "value")
density_xy_coord2$lng_grid = x_grid[density_xy_coord2$Var1]
density_xy_coord2$lat_grid = y_grid[density_xy_coord2$Var2]

# Plot: 2D Gaussian KDE
ggplot(density_xy_coord2, aes(density_xy_coord2$lng_grid, density_xy_coord2$lat_grid)) +
  geom_raster(aes(fill = density_xy_coord2$value), show.legend =  FALSE) +
  scale_fill_distiller(palette = "Spectral", direction = -1) +
  labs(x="end_lng",
       y="end_lat",
       title = bquote("2D KDE with Gaussian"),
       fill = "density") +
  theme(text = element_text(family = 'Fira Sans'),
        plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank(),
        panel.background = element_rect(fill = "transparent"))


## Next, let's try to visualize the trips as arrows
## Highlight one trip in red
ggplot() +
  geom_segment(data = data,
               aes(x = start_lng_r, y = start_lat_r, xend = end_lng_r, yend = end_lat_r),
               arrow = arrow(length = unit(0.15, "cm")),
               size = 0.15,
               color = "black") +
  geom_segment(data = data[3,],
               aes(x = start_lng_r, y = start_lat_r, xend = end_lng_r, yend = end_lat_r),
               arrow = arrow(length = unit(0.15, "cm")),
               size = 1.5,
               color = "red") +
  labs(x = "longitude",
       y = "latitude",
       title = "Sep. 2023 trips") +
  theme_minimal() + 
  theme(text = element_text(family = 'Fira Sans'),
        plot.title = element_text(hjust = 0.5))

## Next, let's try to show an indicator kernel using
## our custom distance metric

# Function to calculate the absolute value of the
# sine of the angle between two 2D vectors
sin_theta_2d <- function(u, v) {
  if (length(u) != 2 || length(v) != 2) {
    stop("Both vectors must be 2-dimensional")
  }
  cross_product <- u[1] * v[2] - u[2] * v[1]
  return(abs(cross_product) / (norm(u, type = "2") * norm(v, type = "2")))
}

# Form the two components of the trips as 2D vectors
# (emanating from the origin)
data$u1 <- data$end_lng_r - data$start_lng_r
data$u2 <- data$end_lat_r - data$start_lat_r
data$zeros <- 0

# Let's make sure that worked
# Should show all trips emanating from the origin
ggplot() +
  geom_segment(data = data,
               aes(x = zeros, y = zeros, xend = u1, yend = u2),
               arrow = arrow(length = unit(0.15, "cm")),
               size = 0.15,
               color = "black") +
  labs(x = "change in lon",
       y = "change in lat",
       title = "As vectors from origin") +
  geom_segment(data = data[3,],
               aes(x = zeros, y = zeros, xend = u1, yend = u2),
               arrow = arrow(length = unit(0.15, "cm")),
               size = 1.5,
               color = "red") +
  theme_minimal() + 
  theme(text = element_text(family = 'Fira Sans'),
        plot.title = element_text(hjust = 0.5))

# Now check if the sin_theta_2d function works
# with two random trips
i <- 4
sin_theta_2d(c(data$u1[i], data$u2[i]), c(data$u1[i+2], data$u2[i+2]))

# Subset just the relevant spatial columns
spatial_data <- data %>% select(start_lng_r, start_lat_r, end_lng_r, end_lat_r, u1, u2)

# Custom spatial distance function

# Find max start or end dist first
start_dists <- function(x, y) {
  return(sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)[1, 1])
}
end_dists <- function(x, y) {
  return(sqrt((x[3] - y[3])^2 + (x[4] - y[4])^2)[1, 1])
}
candidate_maxes <- matrix(0, nrow = N, ncol = 1)
for (i in 1:N) {
  candidate_maxes[i] <- max(start_dists(spatial_data[i,], spatial_data[3,]),
                            end_dists(spatial_data[i,], spatial_data[3,]))
}
true_max <- max(candidate_maxes)

# Actual spatial distance function  
spatial_distance <- function(x, y, gam) {
  start_dist <- sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)[1, 1]
  end_dist <- sqrt((x[3] - y[3])^2 + (x[4] - y[4])^2)[1, 1]
  
  angular_pen <- sin_theta_2d(x[5:6], y[5:6])[1, 1]
  
  return((1 - gam) * (start_dist + end_dist) / (2 * true_max) + gam * angular_pen)
}

## Turn indicator kernel visualization into a function
## so we can call it multiple times more easily
indicator_kernel_visualize <- function(h, gam) {
  # Initialize matrix to hold custom distance from each 
  # trip to the red trip
  gam <- gam # set angular penalty hyper-parameter
  distances_from_red <- matrix(0, nrow = N, ncol = 1)
  
  # Calculate distances from all other trips to the red 
  # trip (trip 3)
  for (i in 1:N) {
    distances_from_red[i] <- spatial_distance(spatial_data[i,], spatial_data[3,], gam)
  }
  
  # Define bandwidth / indicator cut-off for custom 
  # metric space
  h <- h
  
  # Initialize matrix to hold indicators of which trips
  # are "close" to the red trip according to our custom
  # distance metric
  N <- nrow(spatial_data)
  close_to_red <- matrix(0, nrow = N, ncol = 1)
  
  # Calculate indicators of all other trips w.r.t.
  # the red trip (trip 3)
  for (i in 1:N) {
    if (distances_from_red[i] < h) {
      close_to_red[i] <- 1
    }
  }
  print(paste(sum(close_to_red), "trips close to red"))
  
  # Subset the spatial data into those trips that were
  # close to red, and those that were not
  spatial_data$indic <- close_to_red
  close <- spatial_data %>% filter(indic == 1)
  not_close <- spatial_data %>% filter(indic == 0)
  not_close_distinct <- not_close %>% distinct()
  
  # Visualize the separation achieved by the indicator
  # kernel using our custom distance metric, for the
  # chosen value of gamma
  plot <- ggplot() +
    geom_segment(data = close,
                 aes(x = start_lng_r, y = start_lat_r, xend = end_lng_r, yend = end_lat_r),
                 arrow = arrow(length = unit(0.15, "cm")),
                 size = 0.3,
                 color = "black") +
    geom_segment(data = not_close_distinct,
                 aes(x = start_lng_r, y = start_lat_r, xend = end_lng_r, yend = end_lat_r),
                 size = 0.1,
                 color = "black",
                 alpha = 0.1) +
    geom_segment(data = data[3,],
                 aes(x = start_lng_r, y = start_lat_r, xend = end_lng_r, yend = end_lat_r),
                 arrow = arrow(length = unit(0.3, "cm")),
                 size = 4,
                 color = "red") +
    labs(x = "longitude",
         y = "latitude",
         title = bquote("h" == .(h) ~ ",  gamma" == .(gam))) +
    theme_minimal() + 
    theme(text = element_text(family = 'Fira Sans'),
          plot.title = element_text(hjust = 0.5, size = 30),
          axis.title = element_text(size = 20),
          axis.ticks = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          panel.background = element_rect(fill = "white", color = NA),  # Set panel background to white
          plot.background = element_rect(fill = "white", color = NA))    # Set plot background to white)
  
  return(plot)
}

# Save plots for a small grid of hyper-parameter values
h_values <- c(0.2, 0.35)
gamma_values <- c(0.1, 0.5, 0.9)

for (h in h_values) {
  for (gam in gamma_values) {
    # Create plot
    plot <- indicator_kernel_visualize(h, gam)
    
    # Save plot
    filename <- paste0("h_", h, "_gam_", gam, ".png")
    ggsave(filename, plot, width = 8, height = 6)
  }
}
