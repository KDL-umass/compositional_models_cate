We use publicly available [Stack Overflow database](https://archive.org/details/stackexchange) and [user-generated queries](https://data.stackexchange.com/stackoverflow/queries?order_by=popular) to collect query executions data.

**Note:** We follow a bit complicated process to setup StackOverflow (SO) database. This is because SO data is SQL Server based and our goal is to use Python-based Postgres packages to query Postgres database. So, first we create a SQL Server database and then we do Postgres migration. There might be a simpler way to directly setup Postgres database for SO data or better compatibility between Python and  SQL server querying for MAC OS. We describe the slight complicated process below. 

### Download StackOverflow data. 

This is a nice [user guide](https://www.brentozar.com/archive/2015/10/how-to-download-the-stack-overflow-database-via-bittorrent/) to follow.

- Assume data is downloaded in this location: <path-to-so-database-dir>/StackOverflow2013*.mdf

As Stack Overflow database is SQL server based which is windows only compatible. We use Docker to setup database and query from it. 
Steps to setup StackOverflow Database in SQL Server in MAC OS
- Install [Docker for MACOS](https://docs.docker.com/desktop/setup/install/mac-install/).
- Install SQL Server
    - Pull Docker Image: https://setapp.com/how-to/install-sql-server
    ``` 
    docker pull mcr.microsoft.com/mssql/server:2019-latest 
    ```
    - Launch SQL Server 
    ```docker run --name SQLServer -v <path-to-so-database-dir>:/var/opt/mssql/data2 -e "ACCEPT_EULA=Y" 
    -e "SA_PASSWORD=<password>" -p 1433:1433 -d mcr.microsoft.com/mssql/server:2019-latest```


- Attach data to the database 
``` SQL
USE [master]
GO
CREATE DATABASE [StackOverflow2013] ON
(FILENAME = N'/var/opt/mssql/data2/StackOverflow2013_1.mdf'),
(FILENAME = N'/var/opt/mssql/data2/StackOverflow2013_2.ndf'),
(FILENAME = N'/var/opt/mssql/data2/StackOverflow2013_3.ndf'),
(FILENAME = N'/var/opt/mssql/data2/StackOverflow2013_4.ndf')
LOG ON
(FILENAME = N'/var/opt/mssql/data2/StackOverflow2013_log.ldf')
FOR ATTACH;
``` 
- Connect to the database in Azure Data studioRun a sample query in MS SQL Server.

### Postgres Migration
- Install posgresql: ```brew install postgresql@14```
- If it's already installed, to start postgresql: ```brew services start postgresql@14```
- ```psql postgres```
- Create database: ```create database stackoverflow2013 owner <ownername>;```
- Connect to database: ```\c stackoverflow <ownername>;```
- Create extension: ```Create extension "uuid-ossp"```

- TODO: Add steps for data migration 

### Generate query executions for different database interventions 
Run bash script ```./end_to_end_data_gen.sh``` 
Above script pulls public user queries from SO, parameterizes the query, does pre-processing, execution and post-processing, resulting in storing query execution plans for different database interventions. It creates data in both json and CSV format for modeling.


