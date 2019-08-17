#İSTATİSTİKSEL TESTLER

## VERİSETİMİZİ OLUŞTURALIM
veri <- rnorm(100, 40, 10)
numara <- c(1:100) 

## HİSTOGRAM VE GGQQPLOT(HARİCİ KÜTÜPHANE) İLE GRAFİĞİNE BAKALIM
hist(veri, density = 100, col = "turquoise4")

### harici kütphane ###
install.packages("ggpubr")

library(ggpubr)

ggqqplot(veri)


## T- TESTİ
### SHAPİRO-WİLKS TESTİ
shapiro.test(veri)

## R içindeki t-testi ile bakalım
t.test(veri, mu = 120, alternative = "two.sided", conf.level = 0.95)

### alternatif kütüphane ile bakalım
install.packages("inferr")
library(inferr)

#### verisetini dataframe'e çevirmemiz lazı kullanabilmek için
df <- as.data.frame(veri)
df
infer_os_t_test(df, veri, mu = 120, alpha = 0.05, alternative = "all")

#dağılım normal değilse t-testi örneklem 30'dan büyük olduğunda yine kullanılır
#ama 30'dan az ise şöyle bir kütüphane ve fonksiyon kullanabiliriz.

install.packages("DescTools")
library(DescTools)

SignTest(df$veri, mu = 120)
#bunun farkı dağılımk normal değilse medyanı kullanıyor olması

#Tek örneklem oran testi
##eğer bir verisetinde oran değerini test etmek istersek kullanılır

library(tidyverse)

#bunu saçma şekilde yaptım.
infer_os_prop_test(df, variable = veri, prob = 0.5, phat = 0.6, alternative = "less")


# A-B TESTİ (BAĞIMSIZ İKİ ÖRNEKLEM T-TESTİ)

## Veriseti oluşturalım

veri <- data.frame(
  A = rnorm(100, mean = 18000, sd = 5000),
  B = rnorm(100, mean = 20000, sd = 4500)
)

#aynı veriyi farklı şekilde yapalım.
A <- data.frame(degerler=veri$A, sinif = "A")
B <- data.frame(degerler = veri$B, sinif="B")
yeni_veri <- rbind(A,B)
yeni_veri

#küçük bir inceleme yapalım
library(funModeling)
profiling_num(veri)


#grafiksel incelemelerle test yapalım
ggplot(veri, aes(A, B)) + geom_boxplot()



ggplot(yeni_veri, aes(sinif, degerler, fill=sinif)) + geom_boxplot()

ggplot(yeni_veri, aes(degerler, fill=sinif)) + 
  geom_histogram(aes(y=..density..), bins = 100) +
  geom_density(alpha=0.3) +
  facet_grid(sinif~.)

# numerik istatistiksel test yapılması

apply(veri, 2, shapiro.test)

#varyansları farklı mı test edelim (Varyans Homojenliği Testi)
install.packages("car")
library(car)

leveneTest(yeni_veri$degerler~yeni_veri$sinif, mean = center)

t.test(yeni_veri$degerler~yeni_veri$sinif, var.equal=TRUE)


#alternatif fonksiyon
library(inferr)
infer_ts_ind_ttest(data = yeni_veri, x = sinif, y = degerler, 
                   confint = 0.95, alternative = "all")


# BAĞIMLI İKİ ÖRNEKLEM T-TESTİ

### iki örneklem birbirine bağımlıysa veya aynıysa, örneğin öncesi ve 
### sonrası için aynı çalışanları bir eğitimden sonra değerlendirmek
### istiyorsak bağımlı iki örneklem t testi kullanılmalıdır.

## verisetini oluşturalım.
### Normal dağılıma sahip iki veriseti oluşturalım
oncesi <- rnorm(40, mean = 140, sd = 25)
sonrasi <- rnorm(40, mean = 155, sd = 30)

#verileri data frame haline getirip etiketleyelim
A <- data.frame(Ort_Satis=oncesi, Durum = "önce")
B <- data.frame(Ort_Satis=sonrasi, Durum = "sonra")

#A ve B'yi birleştirelim
veriseti = rbind(A,B)
veriseti

library(ggplot2)
library(tidyverse)
library(funModeling)
library(dplyr)


#ilk başta grafiklere bakalım
ggplot(veriseti, aes(Durum,Ort_Satis, fill=Durum)) + geom_boxplot() 

ggplot(veriseti, aes(Ort_Satis, fill=Durum)) +
  geom_histogram(aes(y = ..density..)) +
  geom_density(alpha=0.3) +
  facet_grid(Durum~.)

#varsayalım ki veri iç içe geldi. tıpkı bizim verisetimizdeki gibi
#dolayısıyla oncesi ve sonrasını ayırmamız gerekecek.
#o zaman nasıl istatistiklere ulaşacağız.

# pipe kullanalım bu sefer
veriseti%>%
  group_by(Durum)%>%
  summarise(mean(Ort_Satis), sd(Ort_Satis), var(Ort_Satis))

#verisetini aldık, Durum kolonundaki tipe göre grupladık, sonra istatistikleri çektik.

## TESTLERİN UYGULANMASI
#şimdi shapiro-wilk testi uygulayalım
apply(data.frame(oncesi, sonrasi), MARGIN = 2, FUN = shapiro.test)
#apply kullanırken FUN= kısmında fonksiyonların parantezleri yazılmaz 
# shapiro testinin hipotezi örneklem dağılımı ile anakitle dapılımı arasında fark olmadığıdır.
# yani verimiz normal dağılmıştır şeklinde bir boş hipotezi var.
# p value 0.05'den büyük çıktığı için boş hipotez reddedilemez. yani verimiz normal dağılmıştır.

# BAĞIMLI İKİ ÖRNEKLEM T TESTİ

t.test(veriseti$Ort_Satis~veriseti$Durum, paired=TRUE) #paired bağımlı demek.
#bu ilk testin sonucunda p-value = 0.072 çıktı. yani boş hipotez reddedilemez.
#yani, iki veriseti arasında anlamlı bir farklılık yok.
#sonuç olarak eğitimlerden sonra personelimizin performansında istatistiki bir 
#artış yok. Çalışmamız ortaya anlamlı bir sonuç çıkaramamış ve başarısız olmuş.


#2. bir alternatid fonksiyon daha deneyelim
df <- data.frame(oncesi, sonrasi)
df
library(inferr)
infer_ts_paired_ttest(data=df, x = oncesi, y = sonrasi, confint=0.95, 
             alternative="all")
#yine burada da aynı sonuçlar çıktı
#en altta ~= hipotezi görüldüğü gibi reddedilemiyor. yani fark yok.
#sonuç olarak çalışanlara eğitim verdikten sonra bir fark oluşmamış

# Aynı veriler için non-parametrik test
## non-parametrik test normallik varsayımı sağlanmadığında uygulanır 
wilcox.test(df$oncesi, df$sonrasi, paired = T)


# İKİ ÖRNEKLEM ORAN TESTİ

## iki tane butona ilişkin görüntülenme ve tıklanma oranları olsun
## buna göre bir dönüşüm oranı belirleyip anlamlı bir fark var mı bakalım.

## kırmızı buton 300 tıklanma - 1000 görüntülenme
## yeşil buton 250 tıklanma - 1100 görüntülenme

df <- data.frame(görüntülenme=c(1000,1100),
                 tıklanma=c(300,250), row.names = c("kırmızı", "yeşil"))


# testin yapılması
library(inferr)
infer_ts_prop_test(data = df, görüntülenme, tıklanma)
#alternatif fonksiyon
install.packages("mosaic")
library(mosaic)
mosaic::prop.test(x=c(300,250), n=c(1000, 1100))
## p-value 0.05'ten küçük ve sonuç olarak anlamlı bir fark vardır diyoruz.


# ANOVA TESTİ - VARYANS ANALİZİ

## İkiden fazla grup olduğunda varyans analizi kullanılır.

# siteye 3 farklı tema uygulayıp anasayfada kalma sürelerini inceleyelim

A <- rnbinom(100, 1000, prob = 0.5)
B <- rnbinom(100, 1000, prob = 0.5)
C <- rnbinom(100, 1000, prob = 0.52)
hist(A)
hist(B)
hist(C)

dfa <- data.frame(tür ="A", süre = A)
dfb <- data.frame(tür ="B", süre = B)
dfc <- data.frame(tür ="C", süre = C)

df <- rbind(dfc,dfb, dfa)

df
library(ggplot2)

ggplot(df, aes(tür, süre, fill=süre)) + 
  geom_boxplot()
  

ggplot(df, aes(süre, fill=tür)) + geom_histogram()

ggplot(df, aes(süre, fill=tür)) + 
  geom_histogram(aes(y=..density..), bins = 100) +
  geom_density(alpha=0.3) +
  facet_grid(tür~.)

library(dplyr)
#pipe kullanacağız ve dplyr içinde bulunuyor pipe lar
group_by(df, tür) %>%
  summarise(mean(süre), median(süre), sd(süre))


## testler

bartlett.test(süre~tür, data = df)
library(DescTools)
DescTools::LeveneTest(süre~tür, data =df)


# normal dağılıma sahip miyiz shapiro.test ile bakalım

# A, B, C gruplarında süre normal dağılmış mı bakalım.
shapiro.test(df[df$tür=="A", ]$süre)
shapiro.test(df[df$tür=="B", ]$süre)
shapiro.test(df[df$tür=="C", ]$süre)

#her üçünde de süre normal dağılmış.
# p değeri 0.05'ten büyük ve boş hipotez reddedilmez

# ANOVA HİPOTEZ TESTİ
anova = aov(süre~tür, data = df)

summary(anova)

# p-value değeri 0.05'ten küçük olduğu için h0 reddedilir.
# yani grupların varyansı birbirinden farklıdır.


#şimdi bu farklılığın nereden kaynaklandığını bulalım
TukeyHSD(anova)

# alternatif fonksiyon
library(inferr)
infer_oneway_anova(df, süre, tür)

#non-parametrik test
kruskal.test(süre~tür, data = df)


# KORELASYON TESTLERİ

### bütün verilerimizi silelim
rm(list = ls())

## SATICILARIN SKORU İLE YAPTIKLARI SATIŞLAR ARASINDA 
## BİR KORELASON VEYA ANLAMLI BİR İLİŞKİ VAR MI? İNCELEYELİM.

### verisetinin oluşturulması
df <- mtcars

library(ggpubr)
ggscatter(df, x = "mpg", y ="wt", add = "reg.line", conf.int = T,
          cor.coef = T,
          cor.method = "pearson")

shapiro.test(df$mpg)
shapiro.test(df$wt)

cor.test(df$wt, df$mpg, method = "pearson")

#eğer varsayımlar sağlanmıyorsa yani veriseti normal dağılıma sahip değilse
# method olarak non-parametrik yöntem olan spearman kullanılabilir.

cor.test(df$wt, df$mpg, method = "spearman")
cor.test(df$wt, df$mpg, method = "kendall")

#alternatif yöntemler
cor(df, use = "complete.obs")
library(Hmisc)
rcorr(as.matrix(df))

ggscatter(df, x = "qsec", y ="wt", add = "reg.line", conf.int = T,
          cor.coef = T,
          cor.method = "pearson")

# GELİŞMİŞ KORELASYON MATRİSİ GRAFİĞİ
library(PerformanceAnalytics)

df <- mtcars[, c(1,3,4,5,6,7)]
chart.Correlation(df, histogram = T, pch=19, method = "pearson")
