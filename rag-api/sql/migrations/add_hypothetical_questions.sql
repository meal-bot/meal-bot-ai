-- recipes.hypothetical_questions JSON NULL 컬럼 추가 (멱등)
-- MySQL 8.0 호환: ADD COLUMN IF NOT EXISTS 대신 information_schema + PREPARE 사용

SET @col_exists := (
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'recipes'
      AND column_name = 'hypothetical_questions'
);

SET @stmt := IF(
    @col_exists = 0,
    "ALTER TABLE recipes ADD COLUMN hypothetical_questions JSON NULL COMMENT 'GPT 생성 가상 사용자 질문 3개 (검색용)'",
    "SELECT 'hypothetical_questions 컬럼이 이미 존재합니다.' AS notice"
);

PREPARE stmt FROM @stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
