#installing necessary packages

install.packages(c("distill", "rmarkdown", "postcards"))

#checking package versions

packageVersion("distill")
packageVersion("rmarkdown")
packageVersion("postcards")

library(distill)
create_website(dir = ".", title = "mfeo", gh_pages = TRUE)

distill::create_article

create_article(file = "Maura Chiles",
               template = "jolla",
               package = "postcards")