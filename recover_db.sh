gunzip -k dbbackup/dump.sql.gz
PGPASSWORD=dai psql -h 0.0.0.0 -p 5432 -U postgres -f dbbackup/dump.sql
rm -rf dbbackup/dump.sql


