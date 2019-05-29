CREATE TABLE IF NOT EXISTS daily(stock_id VARCHAR(10) NOT NULL,
                                 time_stamp  timestamp with time zone NOT NULL,
                                 open       FLOAT(4) NOT NULL,
                                 high       FLOAT(4) NOT NULL,
                                 low        FLOAT(4) NOT NULL,
                                 last       FLOAT(4) NOT NULL,
                                 volume     INT NOT NULL,
                                 PRIMARY KEY(stock_id, time_stamp)
                                 );

CREATE INDEX IF NOT EXISTS daily_time_stamp_idx
  ON daily(time_stamp);

CREATE TABLE IF NOT EXISTS minute(stock_id VARCHAR(10) NOT NULL,
                                 time_stamp  timestamp with time zone NOT NULL,
                                 open FLOAT(4) NOT NULL,
                                 high FLOAT(4) NOT NULL,
                                 low  FLOAT(4) NOT NULL,
                                 last FLOAT(4) NOT NULL,
                                 volume INT NOT NULL,
                                 PRIMARY KEY(stock_id, time_stamp)
                                 );

CREATE INDEX IF NOT EXISTS minute_time_stamp_idx
  ON minute(time_stamp);


CREATE TABLE IF NOT EXISTS stocks(stock_id VARCHAR(10) NOT NULL,
                                  stock_name VARCHAR(10) NOT NULL
                                 );

INSERT INTO stocks(stock_id, stock_name) VALUES(3966,    'ABB');
INSERT INTO stocks(stock_id, stock_name) VALUES(18634,   'ALFA');
INSERT INTO stocks(stock_id, stock_name) VALUES(402,     'ASSA-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(3524,    'AZN');
INSERT INTO stocks(stock_id, stock_name) VALUES(45,      'ATCO-A');
INSERT INTO stocks(stock_id, stock_name) VALUES(46,      'ATCO-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(47,      'ALIV-SDB');
INSERT INTO stocks(stock_id, stock_name) VALUES(15285,   'BOL');
INSERT INTO stocks(stock_id, stock_name) VALUES(81,      'ELUX-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(101,     'ERIC-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(139301,  'ESSITY-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(812,     'GETI-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(992,     'HM-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(812,     'GETI-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(819,     'HEXA-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(161,     'INVE-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(999,     'KINV-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(160271,  'NDA-SE');
INSERT INTO stocks(stock_id, stock_name) VALUES(4928,    'SAND');
INSERT INTO stocks(stock_id, stock_name) VALUES(323,     'SCA-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(281,     'SEB-A');
INSERT INTO stocks(stock_id, stock_name) VALUES(401,     'SECU-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(340,     'SHB-A');
INSERT INTO stocks(stock_id, stock_name) VALUES(283,     'SKA-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(285,     'SKF-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(300,     'SSAB-A');
INSERT INTO stocks(stock_id, stock_name) VALUES(120,     'SWED-A');
INSERT INTO stocks(stock_id, stock_name) VALUES(361,     'SWMA');
INSERT INTO stocks(stock_id, stock_name) VALUES(1027,    'TEL2-B');
INSERT INTO stocks(stock_id, stock_name) VALUES(5095,    'TELIA');
INSERT INTO stocks(stock_id, stock_name) VALUES(366,     'VOLV-B');
