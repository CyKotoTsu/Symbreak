library(ggplot2)
library(ggh4x)
library(tibble)
library(reticulate)
library(viridisLite)
library(viridis)
library(rgl)
library(akima)
library(metR)
library(ggvfields)
library(Hmisc)
library(MASS)

# Preliminaries-----------------------------------------------------------------

# Define a custom global theme
theme_cy <- function(base_size = 12, base_family = "Latin Modern math") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Text
      plot.title = element_text(face = "bold", size = base_size, hjust = 0.5),
      plot.subtitle = element_text(size = base_size, hjust = 0.5, color = "grey40"),
      plot.caption = element_text(size = base_size, hjust = 1, color = "grey40"),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text.x = element_text(size = base_size, color = "grey40"),
      axis.text.y = element_text(size = base_size, color = "grey40"),
      axis.ticks = element_line(color = "grey40", linewidth = 0.4), # draw ticks
      axis.ticks.length = unit(5, "pt"),                          # tick length
      
      # Grid and axes
      panel.grid.major = element_line(color = "grey85", size = 0.3),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "gray40", linewidth = 0.5),
      
      # Legend
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      legend.background = element_blank(),
      
      # Background and margins
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      strip.background = element_rect(fill = "grey90", color = NA),
      strip.text = element_text(face = "bold")
    )
}

# Apply the theme globally
theme_set(theme_cy())
#Set working directory
setwd("/Users/chengyouyu/Desktop/Gastrulation/SymBreak/csv_outputs/bilateral")
}


cent<-read.table('RT_centroids_mod.csv',header=T,sep=',')

dve_df<- cent[cent$type=='DVE',]
spot_df<- cent[cent$type=='spot',]

angle_df<-data.frame(angle=rep(0,20))
for(i in 1:length(dve_df[,1])){
  vec1<-c(dve_df$cx[i],dve_df$cy[i])
  mag1<-sqrt(vec1[1]^2+vec1[2]^2)
  vec2<-c(spot_df$cx[i],spot_df$cy[i])
  mag2<-sqrt(vec2[1]^2+vec2[2]^2)
  dot_prod<- sum(vec1*vec2)
  cos_angle<- dot_prod/(mag1*mag2)
  theta<-acos(cos_angle)
  angle_df$angle[i]<-theta
}


# Generate ellipse points
t <- seq(0, 2*pi, length.out = 500)
df <- data.frame(
  x = 10.46 * cos(t),
  y = 7.46 * sin(t)
)

ggplot(cent,aes(cx,cy,group=type,color=as.character(sim)))+
  geom_point(aes(color= sim, shape=type),size=2.5,show.legend=T)+
  geom_line(aes(group = sim, color = sim),lty=3,show.legend=F)+
  scale_color_stepsn(n.breaks = 20, colours = viridis::turbo(9))+
  labs(shape = "",x=expression(paste('x-position')),y=expression(paste('y-position')))+
  theme(axis.line = element_line(color = NA),
        panel.border = element_rect(colour = "white", fill=NA, linewidth=0.5),
        legend.position = 'none')+
  scale_shape_manual(
    values = c("DVE" = 16, "spot" = 8),   # adapt to your levels
    labels = c("DVE", "Organiser")
  )+
  geom_path(data=df,aes(x,y),linewidth=0.3,inherit.aes=FALSE)+
  scale_x_continuous(limits=c(-11,11))+
  scale_y_continuous(limits=c(-11,11))+
  force_panelsizes(rows = unit(35, "mm"),
                   cols = unit(35,"mm"))
  

tracking<-read.csv('RT_track_centroids_mod.csv',header=T,sep=',')

ggplot(tracking, aes(x = cx, y = cy, group = as.character(sim), color = as.character(sim)))+
  geom_path(arrow = arrow(type = "open", length = unit(0.05, "inches"))) +
  coord_fixed()+
  scale_x_continuous(limits=c(-11,11))+
  scale_y_continuous(limits=c(-11,11))+
  geom_path(data=df,aes(x,y),linewidth=0.3,inherit.aes=FALSE)+
  theme(legend.position='none')+
  labs(x='x-position',y='y-position')+
  force_panelsizes(rows = unit(35, "mm"),
                   cols = unit(35,"mm"))
