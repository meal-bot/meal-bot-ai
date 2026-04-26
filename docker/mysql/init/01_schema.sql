CREATE TABLE IF NOT EXISTS recipe (
    rcp_seq     INT          NOT NULL COMMENT '레시피 고유번호',
    name        VARCHAR(200) NOT NULL COMMENT '레시피명',
    category    VARCHAR(50)           COMMENT '요리 종류 (반찬/국&찌개 등)',
    cooking_way VARCHAR(50)           COMMENT '조리 방법 (찌기/끓이기 등)',
    ingredients TEXT                  COMMENT '재료 원본 텍스트',
    hash_tag    VARCHAR(200)          COMMENT '해시태그',
    calories    INT                   COMMENT '열량 (kcal)',
    carbs       INT                   COMMENT '탄수화물 (g)',
    protein     INT                   COMMENT '단백질 (g)',
    fat         INT                   COMMENT '지방 (g)',
    sodium      INT                   COMMENT '나트륨 (mg)',
    img_main    VARCHAR(500)          COMMENT '대표 이미지 URL',
    img_thumb   VARCHAR(500)          COMMENT '썸네일 이미지 URL',
    manuals     JSON                  COMMENT '조리 순서 배열',
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (rcp_seq),
    INDEX idx_category    (category),
    INDEX idx_cooking_way (cooking_way),
    INDEX idx_calories    (calories),
    INDEX idx_sodium      (sodium)
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = '식약처 COOKRCP01 레시피 데이터';
