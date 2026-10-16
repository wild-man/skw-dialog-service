use skw_lib_shared::AppError;
use skw_lib_shared::prelude::migrations::run_migrations;

#[tokio::main]
async fn main() -> Result<(), AppError> {
    run_migrations().await
}
