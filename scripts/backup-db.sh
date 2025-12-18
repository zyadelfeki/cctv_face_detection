#!/usr/bin/env bash
# Database Backup Script
# Usage: ./scripts/backup-db.sh

set -euo pipefail

# Configuration
NAMESPACE="cctv-detection"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS=${RETENTION_DAYS:-30}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/postgres_backup_${TIMESTAMP}.sql.gz"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting database backup...${NC}"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Get PostgreSQL pod
POSTGRES_POD=$(kubectl get pod -n ${NAMESPACE} -l app=postgres -o jsonpath='{.items[0].metadata.name}')

if [ -z "${POSTGRES_POD}" ]; then
    echo -e "${RED}Error: PostgreSQL pod not found${NC}"
    exit 1
fi

echo "Found PostgreSQL pod: ${POSTGRES_POD}"

# Perform backup
echo "Creating backup..."
kubectl exec -n ${NAMESPACE} ${POSTGRES_POD} -- pg_dumpall -U postgres | gzip > ${BACKUP_FILE}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Backup created: ${BACKUP_FILE}${NC}"
    echo "Backup size: $(du -h ${BACKUP_FILE} | cut -f1)"
else
    echo -e "${RED}✗ Backup failed${NC}"
    exit 1
fi

# Cleanup old backups
echo "Cleaning up old backups (older than ${RETENTION_DAYS} days)..."
find ${BACKUP_DIR} -name "postgres_backup_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete

echo -e "${GREEN}✓ Backup complete!${NC}"
