Copy (SELECT * FROM minute ORDER BY stock_id, time_stamp) To '/tmp/data.csv' With (FORMAT CSV, HEADER, DELIMITER ',');
Copy (SELECT * FROM daily ORDER BY stock_id, time_stamp) To '/tmp/data_daily.csv' With (FORMAT CSV, HEADER, DELIMITER ',');
